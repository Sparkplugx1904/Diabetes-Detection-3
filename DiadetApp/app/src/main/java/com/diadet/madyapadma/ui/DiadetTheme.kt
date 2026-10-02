package com.diadet.madyapadma.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Diadet brand: Blue tones (diabetes awareness = blue)
private val LightColors = lightColorScheme(
    primary   = Color(0xFF1565C0),   // Deep blue
    secondary = Color(0xFF03DAC6),
    tertiary  = Color(0xFFBBDEFB),   // Light blue
    error     = Color(0xFFB00020),
)

private val DarkColors = darkColorScheme(
    primary   = Color(0xFF90CAF9),   // Light blue for dark
    secondary = Color(0xFF03DAC6),
    tertiary  = Color(0xFFBBDEFB),
    error     = Color(0xFFCF6679),
)

@Composable
fun DiadetTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColors else LightColors
    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
