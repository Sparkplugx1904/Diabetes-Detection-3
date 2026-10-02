package com.diadet.madyapadma

import android.os.Bundle
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import com.diadet.madyapadma.ui.DiadetNavGraph
import com.diadet.madyapadma.ui.DiadetTheme

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            DiadetTheme {
                DiadetNavGraph()
            }
        }
    }
}
