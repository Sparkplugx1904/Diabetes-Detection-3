package com.diadet.madyapadma

import android.app.Application
import android.util.Log

class DiadetApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Log.i("DiadetApp", "Application started")
    }
}
