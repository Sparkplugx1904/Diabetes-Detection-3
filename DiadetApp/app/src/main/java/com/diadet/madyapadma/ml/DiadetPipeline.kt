package com.diadet.madyapadma.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.diadet.madyapadma.model.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.max

/**
 * DiadetPipeline: Orkestrasi deteksi diabetes dari foto lidah.
 *
 * Alur:
 * 1. Load & decode gambar
 * 2. Jalankan YOLO detection via DiadetDetector (PyTorch Mobile .pt)
 * 3. Ambil deteksi dengan confidence tertinggi
 * 4. Buat annotated bitmap dengan bounding box
 * 5. Return PredictionResult
 */
class DiadetPipeline(private val context: Context) {

    internal val detector = DiadetDetector(context)

    companion object {
        private const val TAG = "DiadetPipeline"
    }

    suspend fun initialize() {
        detector.initialize()
    }

    suspend fun analyze(imagePath: String): PredictionResult = withContext(Dispatchers.Default) {
        val startTime = System.currentTimeMillis()

        var original: Bitmap? = null
        var annotated: Bitmap? = null

        try {
            // 1. Decode image
            original = BitmapFactory.decodeFile(imagePath)
                ?: return@withContext PredictionResult(
                    isDiabetic = false,
                    diabeticProbability = 0f,
                    nonDiabeticProbability = 0f,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Gagal decode gambar"
                )

            // 2. Deteksi
            if (!detector.isReady()) {
                detector.initialize()
            }

            val detections = detector.detect(original)

            if (detections.isEmpty()) {
                return@withContext PredictionResult(
                    isDiabetic = false,
                    diabeticProbability = 0f,
                    nonDiabeticProbability = 0f,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Lidah tidak terdeteksi. Pastikan lidah terlihat jelas."
                )
            }

            // 3. Ambil deteksi terbaik (conf tertinggi)
            val best = detections.first()
            val cls = best.classId.coerceIn(0, DiadetDetector.CLASS_LABELS.size - 1)
            val label = DiadetDetector.CLASS_LABELS[cls]
            val isDiabetic = cls == 0 // class 0 = Diabetes
            val conf = best.confidence

            // 4. Buat annotated bitmap
            annotated = createAnnotatedBitmap(original, best, label)

            // 5. Distribusi probabilitas: kelas pemenang = conf, lainnya = 1-conf
            val diabeticProb = if (isDiabetic) conf else (1f - conf)
            val normalProb   = if (!isDiabetic) conf else (1f - conf)

            return@withContext PredictionResult(
                isDiabetic = isDiabetic,
                diabeticProbability = diabeticProb,
                nonDiabeticProbability = normalProb,
                inferenceTimeMs = elapsed(startTime),
                detectionBox = best.box,
                annotatedBitmap = annotated,
                classLabel = label
            )

        } catch (e: Exception) {
            annotated?.recycle()
            Log.e(TAG, "analyze error: ${e.message}")
            return@withContext PredictionResult(
                isDiabetic = false,
                diabeticProbability = 0f,
                nonDiabeticProbability = 0f,
                inferenceTimeMs = elapsed(startTime),
                error = e.message ?: "Unknown error"
            )
        } finally {
            original?.recycle()
        }
    }

    /**
     * Gambar bounding box dan label pada copy bitmap original.
     */
    private fun createAnnotatedBitmap(original: Bitmap, det: DetectionBox, label: String): Bitmap {
        val copy = original.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(copy)
        val strokeWidth = max(3f, copy.width / 300f)

        val isDiabetic = det.classId == 0
        val boxColor = if (isDiabetic) Color.rgb(229, 57, 53) else Color.rgb(67, 160, 71)

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            color = boxColor
            this.strokeWidth = strokeWidth
        }
        val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.argb(40, Color.red(boxColor), Color.green(boxColor), Color.blue(boxColor))
        }
        val textBgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = boxColor
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(24f, copy.width / 25f)
            isFakeBoldText = true
        }

        val box = det.box
        canvas.drawRect(box, fillPaint)
        canvas.drawRect(box, boxPaint)

        val text = "$label ${"%.1f".format(det.confidence * 100)}%"
        val textH = textPaint.textSize + 8f
        val textBgRect = RectF(box.left, box.top - textH, box.left + textPaint.measureText(text) + 16f, box.top)
        canvas.drawRect(textBgRect, textBgPaint)
        canvas.drawText(text, box.left + 8f, box.top - 6f, textPaint)

        return copy
    }

    private fun elapsed(start: Long) = System.currentTimeMillis() - start

    fun close() {
        detector.close()
    }
}
