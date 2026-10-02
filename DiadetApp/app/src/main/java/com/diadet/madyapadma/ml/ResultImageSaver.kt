package com.diadet.madyapadma.ml

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import com.diadet.madyapadma.model.PredictionResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ResultImageSaver {

    private const val TAG = "ResultImageSaver"

    suspend fun saveResultImage(
        context: Context,
        original: Bitmap,
        result: PredictionResult
    ): Boolean = withContext(Dispatchers.IO) {
        try {
            val bitmapToSave = result.annotatedBitmap ?: original
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val label = if (result.isDiabetic) "Diabetes" else "Normal"
            val filename = "Diadet_${label}_${timestamp}.jpg"

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val values = ContentValues().apply {
                    put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                    put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
                    put(MediaStore.Images.Media.RELATIVE_PATH, "${Environment.DIRECTORY_PICTURES}/Diadet")
                }
                val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                    ?: return@withContext false
                context.contentResolver.openOutputStream(uri)?.use { out ->
                    bitmapToSave.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
            } else {
                val dir = File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "Diadet")
                dir.mkdirs()
                val file = File(dir, filename)
                FileOutputStream(file).use { out ->
                    bitmapToSave.compress(Bitmap.CompressFormat.JPEG, 95, out)
                }
            }
            Log.i(TAG, "Saved: $filename")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Save failed: ${e.message}")
            false
        }
    }
}
