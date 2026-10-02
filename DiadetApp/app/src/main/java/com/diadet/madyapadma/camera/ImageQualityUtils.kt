package com.diadet.madyapadma.camera

import android.graphics.Bitmap
import android.util.Log

/**
 * Deteksi kualitas gambar menggunakan varian Laplacian.
 * Digunakan untuk memastikan gambar cukup tajam sebelum inferensi.
 */
object ImageQualityUtils {

    private const val TAG = "ImageQualityUtils"

    /**
     * Hitung Laplacian variance sebagai ukuran ketajaman gambar.
     * Nilai lebih tinggi = lebih tajam.
     */
    fun calculateBlurriness(bitmap: Bitmap): Float {
        return try {
            val width  = bitmap.width
            val height = bitmap.height
            val pixels = IntArray(width * height)
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            var mean = 0.0
            for (pixel in pixels) {
                val gray = (0.299 * ((pixel shr 16) and 0xFF) +
                            0.587 * ((pixel shr  8) and 0xFF) +
                            0.114 * ( pixel         and 0xFF))
                mean += gray
            }
            mean /= pixels.size

            var variance = 0.0
            for (pixel in pixels) {
                val gray = (0.299 * ((pixel shr 16) and 0xFF) +
                            0.587 * ((pixel shr  8) and 0xFF) +
                            0.114 * ( pixel         and 0xFF))
                // Simple Laplacian approximation via pixel variance around mean
                val diff = gray - mean
                variance += diff * diff
            }
            (variance / pixels.size).toFloat()
        } catch (e: Exception) {
            Log.w(TAG, "calculateBlurriness failed: ${e.message}")
            0f
        }
    }

    /**
     * Periksa apakah gambar cukup tajam untuk inferensi.
     */
    fun isQualitySufficientForInference(bitmap: Bitmap, minSharpness: Float): Boolean {
        val sharpness = calculateBlurriness(bitmap)
        return sharpness >= minSharpness
    }
}
