package com.diadet.madyapadma.ui.components

import android.graphics.BitmapFactory
import android.widget.Toast
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.Warning
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.diadet.madyapadma.R
import com.diadet.madyapadma.camera.CameraViewModel
import com.diadet.madyapadma.model.PredictionResult

@Composable
fun ResultScreen(
    imagePath: String,
    onRetake: () -> Unit,
    viewModel: CameraViewModel = viewModel()
) {
    val predictionResult by viewModel.predictionResult.collectAsStateWithLifecycle()
    val isAnalyzing      by viewModel.isAnalyzing.collectAsStateWithLifecycle()
    val context          = LocalContext.current

    LaunchedEffect(imagePath) {
        viewModel.analyzeImage(imagePath)
    }

    val isLoading = isAnalyzing || (predictionResult == null)

    val scaleAnim by animateFloatAsState(
        targetValue = if (!isLoading) 1f else 0.8f,
        animationSpec = tween(durationMillis = 500),
        label = "scale"
    )

    Surface(modifier = Modifier.fillMaxSize()) {
        when {
            isLoading -> {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(48.dp),
                            color = MaterialTheme.colorScheme.primary
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(stringResource(R.string.analyzing), style = MaterialTheme.typography.bodyLarge)
                    }
                }
            }

            predictionResult != null -> {
                val prediction = predictionResult!!
                if (prediction.error != null) {
                    ErrorContent(message = prediction.error, onRetake = onRetake)
                } else {
                    ResultContent(
                        imagePath = imagePath,
                        prediction = prediction,
                        scaleAnim = scaleAnim,
                        onRetake = onRetake,
                        onSave = {
                            viewModel.saveResultToGallery(imagePath)
                            Toast.makeText(
                                context,
                                context.getString(R.string.saved_to_gallery),
                                Toast.LENGTH_SHORT
                            ).show()
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun ResultContent(
    imagePath: String,
    prediction: PredictionResult,
    scaleAnim: Float,
    onRetake: () -> Unit,
    onSave: () -> Unit
) {
    val bitmap       = remember(imagePath) { BitmapFactory.decodeFile(imagePath) }
    val annotated    = prediction.annotatedBitmap
    val isDiabetic   = prediction.isDiabetic
    val diagColor    = if (isDiabetic) Color(0xFFE53935) else Color(0xFF43A047)
    val diagText     = if (isDiabetic) stringResource(R.string.diabetic) else stringResource(R.string.non_diabetic)
    val isLowConf    = prediction.decisionConfidence < 0.55f

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = stringResource(R.string.diagnosis_result),
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(top = 24.dp)
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Image display — prefer annotated bitmap
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp)
                .clip(RoundedCornerShape(16.dp))
                .scale(scaleAnim)
        ) {
            val displayBitmap = annotated ?: bitmap
            if (displayBitmap != null) {
                Image(
                    bitmap = displayBitmap.asImageBitmap(),
                    contentDescription = "Captured tongue",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            }
        }

        // Low-confidence warning
        if (isLowConf) {
            Spacer(modifier = Modifier.height(12.dp))
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xFFFFA000).copy(alpha = 0.18f))
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Filled.Warning,
                    contentDescription = null,
                    tint = Color(0xFFE65100),
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = stringResource(R.string.low_confidence_warning),
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFFE65100)
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Diagnosis card
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(diagColor.copy(alpha = 0.12f))
                .padding(20.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text(
                    text = diagText,
                    fontSize = 32.sp,
                    fontWeight = FontWeight.ExtraBold,
                    color = diagColor
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "${(prediction.decisionConfidence * 100).format1dp()}%",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    color = diagColor
                )
                if (prediction.classLabel.isNotBlank()) {
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = prediction.classLabel,
                        style = MaterialTheme.typography.bodySmall,
                        color = diagColor.copy(alpha = 0.7f)
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Class probability rows
        Text(
            text = stringResource(R.string.diagnostic_class),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            modifier = Modifier.fillMaxWidth(),
            textAlign = TextAlign.Start
        )
        Spacer(modifier = Modifier.height(8.dp))

        // Hanya kelas pemenang yang ditampilkan — nilai minoritas
        // (100% − confidence) sengaja tidak ditampilkan.
        ClassProbabilityRow(
            label = diagText,
            value = prediction.confidence,
            isWinner = true,
            color = diagColor
        )
        Spacer(modifier = Modifier.height(12.dp))

        // Confidence keputusan diagnosis (dari selisih bukti kelas,
        // bukan sekadar skor mentah kelas pemenang)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = stringResource(R.string.confidence),
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "${(prediction.decisionConfidence * 100).format1dp()}%",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = if (isLowConf) Color(0xFFE65100) else diagColor
            )
        }
        Spacer(modifier = Modifier.height(8.dp))
        ConfidenceBar(
            confidence = prediction.decisionConfidence,
            color = if (isLowConf) Color(0xFFE65100) else diagColor
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = stringResource(R.string.confidence_explainer),
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text = "Inference: ${prediction.inferenceTimeMs}ms",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Spacer(modifier = Modifier.height(20.dp))

        // Medical disclaimer
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f))
                .padding(12.dp)
        ) {
            Text(
                text = stringResource(R.string.medical_disclaimer),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Action buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            OutlinedButton(
                onClick = onRetake,
                modifier = Modifier.weight(1f).height(56.dp),
                shape = RoundedCornerShape(12.dp)
            ) { Text(stringResource(R.string.retake), fontSize = 14.sp) }

            Button(
                onClick = onSave,
                modifier = Modifier.weight(1f).height(56.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary)
            ) {
                Icon(Icons.Filled.Save, contentDescription = null, modifier = Modifier.size(20.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text(stringResource(R.string.save), fontSize = 14.sp)
            }
        }

        Spacer(modifier = Modifier.height(16.dp))
    }
}

@Composable
private fun ClassProbabilityRow(label: String, value: Float, isWinner: Boolean, color: Color) {
    val pct      = (value * 100).coerceIn(0f, 100f)
    val barWidth = (value * 100).coerceIn(2f, 100f)
    val bgColor  = if (isWinner) color.copy(alpha = 0.10f) else Color(0xFFF5F5F5)

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(bgColor)
            .padding(12.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = if (isWinner) FontWeight.Bold else FontWeight.Medium,
                color = if (isWinner) color else MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.weight(1f)
            )
            Text(
                text = "${pct.format1dp()}%",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = if (isWinner) color else MaterialTheme.colorScheme.onSurface
            )
        }
        Spacer(modifier = Modifier.height(6.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp)
                .clip(RoundedCornerShape(4.dp))
                .background(Color(0xFFE0E0E0))
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth(fraction = barWidth / 100f)
                    .height(8.dp)
                    .background(color)
            )
        }
    }
}

@Composable
private fun ConfidenceBar(confidence: Float, color: Color) {
    val pct = (confidence * 100).coerceIn(2f, 100f)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(10.dp)
            .clip(RoundedCornerShape(5.dp))
            .background(Color(0xFFE0E0E0))
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth(fraction = pct / 100f)
                .height(10.dp)
                .background(color)
        )
    }
}

@Composable
private fun ErrorContent(message: String, onRetake: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(
            text = "Detection Failed",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.error
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(24.dp))
        Button(onClick = onRetake) { Text("Try Again") }
    }
}

private fun Float.format1dp(): String = "%.1f".format(this)
