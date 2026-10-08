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

import android.graphics.Matrix
import android.media.ExifInterface

/**
 * DiadetPipeline: Orkestrasi deteksi diabetes dari foto lidah.
 *
 * Alur (sama persis dengan versi web):
 * 1. Load & decode gambar sesuai orientasi EXIF (upright)
 * 2. Jalankan YOLO detection via DiadetDetector (PyTorch Mobile .pt) langsung pada gambar utuh
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
            // 1. Decode gambar dengan orientasi EXIF yang benar
            original = decodeOrientedBitmap(imagePath)
                ?: return@withContext PredictionResult(
                    isDiabetic = false,
                    diabeticProbability = 0f,
                    nonDiabeticProbability = 0f,
                    inferenceTimeMs = elapsed(startTime),
                    error = "Gagal decode gambar"
                )

            // 2. Deteksi langsung dengan model YOLO (sama seperti versi web)
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
                    error = "Lidah tidak terdeteksi. Pastikan lidah terlihat jelas di depan kamera."
                )
            }

            // 3. Ambil deteksi terbaik (conf tertinggi)
            val best = detections.first()
            val cls = best.classId.coerceIn(0, DiadetDetector.CLASS_LABELS.size - 1)
            val label = DiadetDetector.CLASS_LABELS[cls]
            val isDiabetic = cls == 0 // class 0 = Diabetes
            val conf = best.confidence

            // 4. Buat annotated bitmap di atas foto asli
            annotated = createAnnotatedBitmap(original, best, label)

            // 5. Distribusi probabilitas
            val diabeticProb = if (isDiabetic) conf else (1f - conf)
            val normalProb   = if (!isDiabetic) conf else (1f - conf)

            val sWin = if (isDiabetic) best.scoreDiabetes else best.scoreNondiabetes
            val oppOwn = if (isDiabetic) best.scoreNondiabetes else best.scoreDiabetes
            val oppOther = detections.drop(1)
                .filter { it.classId != best.classId }
                .maxOfOrNull { if (isDiabetic) it.scoreNondiabetes else it.scoreDiabetes } ?: 0f
            val opp = maxOf(oppOwn, oppOther)
            val decisionConfidence = (sWin / (sWin + opp)).coerceIn(0f, 1f)

            return@withContext PredictionResult(
                isDiabetic = isDiabetic,
                diabeticProbability = diabeticProb,
                nonDiabeticProbability = normalProb,
                inferenceTimeMs = elapsed(startTime),
                detectionBox = best.box,
                annotatedBitmap = annotated,
                classLabel = label,
                decisionConfidence = decisionConfidence
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
     * Decode file bitmap dengan koreksi rotasi EXIF dari sensor kamera.
     */
    private fun decodeOrientedBitmap(imagePath: String): Bitmap? {
        val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
        return try {
            val exif = ExifInterface(imagePath)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            val rotationDegrees = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            if (rotationDegrees != 0f) {
                val matrix = Matrix().apply { postRotate(rotationDegrees) }
                val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                bitmap.recycle()
                rotated
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.w(TAG, "EXIF rotation check failed: ${e.message}")
            bitmap
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

