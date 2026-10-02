package com.diadet.madyapadma.camera

import android.app.Application
import android.graphics.BitmapFactory
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.diadet.madyapadma.ml.DiadetPipeline
import com.diadet.madyapadma.ml.ResultImageSaver
import com.diadet.madyapadma.model.PredictionResult
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    val pipeline = DiadetPipeline(application)
    val settings = AppSettings(application)

    private val _predictionResult = MutableStateFlow<PredictionResult?>(null)
    val predictionResult: StateFlow<PredictionResult?> = _predictionResult.asStateFlow()

    private val _isAnalyzing = MutableStateFlow(false)
    val isAnalyzing: StateFlow<Boolean> = _isAnalyzing.asStateFlow()

    private val resultLock = Any()

    // Auto-capture
    private val _autoCaptureRequests = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val autoCaptureRequests: SharedFlow<String> = _autoCaptureRequests.asSharedFlow()

    private val _autoCaptureStatus = MutableStateFlow("searching")
    val autoCaptureStatus: StateFlow<String> = _autoCaptureStatus.asStateFlow()

    private val _autoCaptureProgress = MutableStateFlow(0)
    val autoCaptureProgress: StateFlow<Int> = _autoCaptureProgress.asStateFlow()

    // Live detection indicator (whether tongue was found in last frame)
    private val _tongueDetected = MutableStateFlow(false)
    val tongueDetected: StateFlow<Boolean> = _tongueDetected.asStateFlow()

    init {
        viewModelScope.launch {
            try {
                Log.d(TAG, "Warming up PyTorch model...")
                pipeline.initialize()
                Log.d(TAG, "Model warmed up")
            } catch (e: Exception) {
                Log.e(TAG, "Model warmup failed: ${e.message}")
            }
        }
    }

    fun updateTongueDetected(detected: Boolean) {
        _tongueDetected.value = detected
    }

    fun reportAutoCaptureStatus(status: String, progress: Int = 0) {
        _autoCaptureStatus.value = status
        _autoCaptureProgress.value = progress
    }

    fun requestAutoCapture(captureDir: String) {
        viewModelScope.launch {
            _autoCaptureRequests.emit(captureDir)
        }
    }

    fun analyzeImage(imagePath: String) {
        viewModelScope.launch {
            synchronized(resultLock) {
                _predictionResult.value?.annotatedBitmap?.let { bmp ->
                    if (!bmp.isRecycled) bmp.recycle()
                }
                _predictionResult.value = null
            }

            _isAnalyzing.value = true
            pipeline.initialize()
            val result = pipeline.analyze(imagePath)

            synchronized(resultLock) {
                _predictionResult.value = result
            }
            _isAnalyzing.value = false
        }
    }

    fun saveResultToGallery(imagePath: String) {
        val result = _predictionResult.value ?: return
        if (result.error != null) return
        viewModelScope.launch {
            val original = BitmapFactory.decodeFile(imagePath)
            if (original != null) {
                val ok = ResultImageSaver.saveResultImage(getApplication(), original, result)
                Log.d(TAG, "saveResultToGallery: saved=$ok")
                original.recycle()
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        synchronized(resultLock) {
            _predictionResult.value?.annotatedBitmap?.let { bmp ->
                if (!bmp.isRecycled) bmp.recycle()
            }
            _predictionResult.value = null
        }
        pipeline.close()
    }

    companion object {
        private const val TAG = "CameraViewModel"
    }
}
