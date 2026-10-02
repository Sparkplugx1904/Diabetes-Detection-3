package com.diadet.madyapadma.model

import android.graphics.Bitmap
import android.graphics.RectF

/**
 * Hasil analisis diabetes dari foto lidah.
 *
 * - isDiabetic: kelas diagnostik yang dipilih
 * - diabeticProbability: probabilitas kelas diabetes (0..1)
 * - nonDiabeticProbability: probabilitas kelas non-diabetes (0..1)
 * - confidence: max(diabeticProbability, nonDiabeticProbability)
 * - margin: selisih absolut kedua kelas
 * - diagnosisPercent: probabilitas kelas pemenang
 * - detectionBox: bounding box lidah yang terdeteksi (nullable)
 * - annotatedBitmap: gambar dengan anotasi bounding box (nullable)
 */
data class PredictionResult(
    val isDiabetic: Boolean,
    val diabeticProbability: Float,
    val nonDiabeticProbability: Float,
    val inferenceTimeMs: Long,
    val error: String? = null,
    val detectionBox: RectF? = null,
    val annotatedBitmap: Bitmap? = null,
    val classLabel: String = ""
) {
    /** Nilai keyakinan = probabilitas kelas pemenang. */
    val confidence: Float
        get() = maxOf(diabeticProbability, nonDiabeticProbability)

    /** Selisih absolut antara kedua kelas (0..1). */
    val margin: Float
        get() = kotlin.math.abs(diabeticProbability - nonDiabeticProbability)

    /** Probabilitas kelas yang menjadi diagnosis. */
    val diagnosisPercent: Float
        get() = if (isDiabetic) diabeticProbability else nonDiabeticProbability
}
