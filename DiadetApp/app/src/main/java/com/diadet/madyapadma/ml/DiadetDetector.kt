package com.diadet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.diadet.madyapadma.model.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * DiadetDetector: engine inferensi PyTorch Mobile (.pt) untuk deteksi diabetes via foto lidah.
 *
 * Model: YOLOv8 detection (best.pt) yang di-export sebagai TorchScript.
 * Input: RGB bitmap 640×640
 * Output: [1, 300, 6] — (x1, y1, x2, y2, conf, cls) atau [1, 6, 8400] raw YOLO format.
 */
class DiadetDetector(private val context: Context) {

    companion object {
        private const val TAG = "DiadetDetector"
        const val MODEL_ASSET = "best.pt"
        const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f

        // Anti false-positive gates (dikalibrasi via probe_color.py / probe_rgb.py):
        // noise → area 99.8% | solid-gray/checker → area ~100%, tonguepx 0% |
        // skin-blob → area 32%, tonguepx 0% | lidah asli → area 14-38%, tonguepx 58-94%.
        private const val MIN_AREA_FRACTION = 0.05f
        private const val MAX_AREA_FRACTION = 0.90f
        private const val MIN_TONGUE_FRACTION = 0.35f

        // Class labels dari model YOLO diabetes (sinkron dengan m.names best.pt)
        val CLASS_LABELS = arrayOf(
            "Diabetes",    // 0 — Terindikasi diabetes
            "Nondiabetes"  // 1 — Normal / non-diabetes
        )

        // Letterbox fill YOLO standard (114/255)
        // NOTE: TensorImageUtils.bitmapToFloat32Tensor menghitung (pixel/255 - mean)/std,
        // jadi untuk normalisasi /255 pakai MEAN=0, STD=1. (STD=1/255 SALAH → input 0-255 → skor ~0)
        private val MEAN = floatArrayOf(0f, 0f, 0f)
        private val STD  = floatArrayOf(1f, 1f, 1f)
    }

    @Volatile private var module: Module? = null
    private val lock = Any()
    @Volatile private var isWarmedUp = false

    /**
     * Salin asset ke file cache agar bisa di-load oleh PyTorch.
     */
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    suspend fun initialize() {
        if (module != null) return
        withContext(Dispatchers.IO) {
            synchronized(lock) {
                if (module != null) return@synchronized
                try {
                    val modelPath = assetFilePath(MODEL_ASSET)
                    module = Module.load(modelPath)
                    Log.i(TAG, "PyTorch Module loaded from: $modelPath")
                    performWarmup()
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to load model: ${e.message}")
                    throw e
                }
            }
        }
    }

    private fun performWarmup() {
        if (isWarmedUp) return
        val mod = module ?: return
        try {
            Log.i(TAG, "Performing warmup inference...")
            val t0 = System.currentTimeMillis()
            val dummyBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
            val inputTensor = bitmapToInputTensor(dummyBitmap)
            dummyBitmap.recycle()
            mod.forward(IValue.from(inputTensor))
            Log.i(TAG, "Warmup done in ${System.currentTimeMillis() - t0}ms")
            isWarmedUp = true
        } catch (e: Exception) {
            Log.w(TAG, "Warmup failed (non-critical): ${e.message}")
        }
    }

    fun isReady(): Boolean = module != null

    /**
     * Jalankan deteksi pada Bitmap.
     * Return: list DetectionBox yang sudah di-NMS, difilter sesuai confidence.
     */
    fun detect(bitmap: Bitmap): List<DetectionBox> = synchronized(lock) {
        val mod = module ?: return emptyList()

        val (letterboxBmp, scale, padLeft, padTop) = letterbox(bitmap, INPUT_SIZE)
        val inputTensor = bitmapToInputTensor(letterboxBmp)
        letterboxBmp.recycle()

        val t0 = System.currentTimeMillis()
        val rawOutput = mod.forward(IValue.from(inputTensor))
        Log.d(TAG, "Inference: ${System.currentTimeMillis() - t0}ms")

        // YOLO TorchScript mengembalikan single tensor [1, 6, 8400]
        // Bukan tuple — harus dicek terlebih dahulu
        val predTensor: Tensor = when {
            rawOutput.isTensor -> rawOutput.toTensor()
            rawOutput.isTuple  -> {
                val tuple = rawOutput.toTuple()
                if (tuple.isNotEmpty()) tuple[0].toTensor() else return emptyList()
            }
            else -> {
                Log.w(TAG, "Unknown output IValue type")
                return emptyList()
            }
        }

        val detections = parseDetections(predTensor, scale, padLeft, padTop, bitmap.width, bitmap.height)
        Log.d(TAG, "YOLO detections found: ${detections.size}")
        return detections
    }

    /**
     * Saring hasil mentah model agar tidak ada false-positive saat tidak ada lidah.
     * Model YOLO teruji memberi conf tinggi bahkan pada noise (0.97) / gambar acak,
     * sehingga confidence saja tidak cukup. Dua gate independen:
     * 1. Fraksi luas box wajar (5-90% area gambar) — menolak halusinasi selebar frame.
     * 2. Fraksi piksel warna lidah (kemerahan) di dalam box — menolak noise / objek non-lidah.
     */
    private fun filterTongueDetections(
        boxes: List<DetectionBox>,
        original: Bitmap
    ): List<DetectionBox> {
        if (boxes.isEmpty()) return boxes
        val imgArea = (original.width * original.height).toFloat()
        var areaRejected = 0
        var colorRejected = 0
        val kept = boxes.filter { det ->
            val areaFrac = (det.box.width() * det.box.height()) / imgArea
            if (areaFrac < MIN_AREA_FRACTION || areaFrac > MAX_AREA_FRACTION) {
                areaRejected++
                return@filter false
            }
            if (tongueColorFraction(original, det.box) < MIN_TONGUE_FRACTION) {
                colorRejected++
                return@filter false
            }
            true
        }
        if (kept.size != boxes.size) {
            Log.d(TAG, "Tongue gates: kept ${kept.size}/${boxes.size} " +
                    "(areaRejected=$areaRejected, colorRejected=$colorRejected)")
        }
        return kept.sortedByDescending { it.confidence }
    }

    /**
     * Fraksi piksel kemerahan (R dominan, saturasi cukup) di dalam box.
     * Ekuivalen RGB dari aturan HSV (H merah, S>=60, V>=60), tanpa perlu OpenCV.
     */
    private fun tongueColorFraction(bitmap: Bitmap, box: RectF): Float {
        val x0 = box.left.toInt().coerceIn(0, bitmap.width - 1)
        val y0 = box.top.toInt().coerceIn(0, bitmap.height - 1)
        val x1 = box.right.toInt().coerceIn(0, bitmap.width)
        val y1 = box.bottom.toInt().coerceIn(0, bitmap.height)
        val w = x1 - x0
        val h = y1 - y0
        if (w <= 0 || h <= 0) return 0f
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, x0, y0, w, h)
        var tongue = 0
        var total = 0
        var i = 0
        while (i < pixels.size) {
            val p = pixels[i]
            val r = (p shr 16) and 0xFF
            val g = (p shr 8) and 0xFF
            val b = p and 0xFF
            val mx = max(max(r, g), b)
            val mn = min(min(r, g), b)
            if (r >= g && r >= b && (mx - mn) >= 60) tongue++
            total++
            i += 3 // subsample: cukup untuk estimasi fraksi, hemat CPU di live analyzer
        }
        return if (total == 0) 0f else tongue.toFloat() / total
    }


    /**
     * Parse output tensor YOLO dan konversi ke koordinat gambar original.
     * Format: [batch, num_det, 6] → (x1,y1,x2,y2,conf,cls) atau
     *         [batch, 6+cls, anchors] → raw YOLO format
     */
    private fun parseDetections(
        tensor: Tensor,
        scale: Float,
        padLeft: Int,
        padTop: Int,
        origW: Int,
        origH: Int
    ): List<DetectionBox> {
        val shape = tensor.shape()
        val data = tensor.dataAsFloatArray

        val results = mutableListOf<DetectionBox>()

        when {
            // Format NMS-ed: [1, N, 6] - sudah di-NMS oleh model export
            shape.size == 3 && shape[2] == 6L -> {
                val numDet = shape[1].toInt()
                for (i in 0 until numDet) {
                    val base = i * 6
                    val conf = data[base + 4]
                    if (conf < CONF_THRESHOLD) continue
                    val cls = data[base + 5].toInt().coerceIn(0, CLASS_LABELS.size - 1)
                    val x1 = ((data[base + 0] - padLeft) / scale).coerceIn(0f, origW.toFloat())
                    val y1 = ((data[base + 1] - padTop)  / scale).coerceIn(0f, origH.toFloat())
                    val x2 = ((data[base + 2] - padLeft) / scale).coerceIn(0f, origW.toFloat())
                    val y2 = ((data[base + 3] - padTop)  / scale).coerceIn(0f, origH.toFloat())
                    // Format NMS-ed tidak menyimpan skor per-kelas:
                    // anggap seluruh conf milik kelas pemenang.
                    val s0n = if (cls == 0) conf else 0f
                    val s1n = if (cls == 1) conf else 0f
                    results.add(DetectionBox(RectF(x1, y1, x2, y2), conf, cls, s0n, s1n))
                }
            }
            // Format raw YOLO: [1, 4+numCls, 8400]
            shape.size == 3 && shape[1] >= 5L -> {
                val numAnchors = shape[2].toInt()
                val numFeatures = shape[1].toInt()
                val numCls = numFeatures - 4
                val rawBoxes = mutableListOf<DetectionBox>()
                var maxScore = 0f

                for (a in 0 until numAnchors) {
                    var bestConf = 0f
                    var bestCls = 0
                    for (c in 0 until numCls) {
                        val score = data[a + (4 + c) * numAnchors]
                        if (score > maxScore) maxScore = score
                        if (score > bestConf) { bestConf = score; bestCls = c }
                    }
                    if (bestConf < CONF_THRESHOLD) continue

                    // Simpan skor kedua kelas untuk decision-confidence
                    // (indeks 4 = Diabetes, indeks 5 = Nondiabetes).
                    val s0 = if (numFeatures > 4) data[a + 4 * numAnchors] else 0f
                    val s1 = if (numFeatures > 5) data[a + 5 * numAnchors] else 0f

                    // cx,cy,w,h in model input space
                    val cx = data[a + 0 * numAnchors]
                    val cy = data[a + 1 * numAnchors]
                    val w  = data[a + 2 * numAnchors]
                    val h  = data[a + 3 * numAnchors]

                    val x1 = ((cx - w / 2 - padLeft) / scale).coerceIn(0f, origW.toFloat())
                    val y1 = ((cy - h / 2 - padTop)  / scale).coerceIn(0f, origH.toFloat())
                    val x2 = ((cx + w / 2 - padLeft) / scale).coerceIn(0f, origW.toFloat())
                    val y2 = ((cy + h / 2 - padTop)  / scale).coerceIn(0f, origH.toFloat())

                    rawBoxes.add(DetectionBox(RectF(x1, y1, x2, y2), bestConf, bestCls, s0, s1))
                }
                results.addAll(nms(rawBoxes, IOU_THRESHOLD))
                Log.d(TAG, "rawAnchors=$numAnchors maxScore=$maxScore kept=${results.size}")
            }
            else -> Log.w(TAG, "Unknown output shape: ${shape.toList()}")
        }

        return results.sortedByDescending { it.confidence }
    }

    /**
     * Non-Maximum Suppression
     */
    private fun nms(boxes: List<DetectionBox>, iouThreshold: Float): List<DetectionBox> {
        val sorted = boxes.sortedByDescending { it.confidence }
        val kept = mutableListOf<DetectionBox>()
        val suppressed = BooleanArray(sorted.size)

        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (iou(sorted[i].box, sorted[j].box) > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft   = max(a.left,   b.left)
        val interTop    = max(a.top,    b.top)
        val interRight  = min(a.right,  b.right)
        val interBottom = min(a.bottom, b.bottom)
        if (interRight <= interLeft || interBottom <= interTop) return 0f
        val interArea = (interRight - interLeft) * (interBottom - interTop)
        val unionArea = (a.right - a.left) * (a.bottom - a.top) +
                        (b.right - b.left) * (b.bottom - b.top) - interArea
        return if (unionArea <= 0f) 0f else interArea / unionArea
    }

    /**
     * Letterbox: resize dengan padding abu-abu ke INPUT_SIZE x INPUT_SIZE.
     * Return: Triple(bitmap, scale, padLeft, padTop)
     */
    private fun letterbox(src: Bitmap, size: Int): LetterboxResult {
        val scale = min(size.toFloat() / src.width, size.toFloat() / src.height)
        val scaledW = (src.width  * scale).toInt()
        val scaledH = (src.height * scale).toInt()
        val padLeft = (size - scaledW) / 2
        val padTop  = (size - scaledH) / 2

        val result = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val scaled = Bitmap.createScaledBitmap(src, scaledW, scaledH, true)
        canvas.drawBitmap(scaled, padLeft.toFloat(), padTop.toFloat(), null)
        scaled.recycle()

        return LetterboxResult(result, scale, padLeft, padTop)
    }

    /**
     * Konversi Bitmap ke input tensor PyTorch.
     * Format: NCHW float32 [1, 3, H, W] dengan normalisasi /255.
     */
    private fun bitmapToInputTensor(bitmap: Bitmap): Tensor {
        return TensorImageUtils.bitmapToFloat32Tensor(bitmap, MEAN, STD)
    }

    fun close() {
        synchronized(lock) {
            module?.destroy()
            module = null
            isWarmedUp = false
            Log.i(TAG, "Module closed")
        }
    }
}

data class LetterboxResult(
    val bitmap: Bitmap,
    val scale: Float,
    val padLeft: Int,
    val padTop: Int
)

data class DetectionBox(
    val box: RectF,
    val confidence: Float,
    val classId: Int,
    /** Skor mentah model untuk kelas Diabetes (0..1). */
    val scoreDiabetes: Float = 0f,
    /** Skor mentah model untuk kelas Nondiabetes (0..1). */
    val scoreNondiabetes: Float = 0f
)
