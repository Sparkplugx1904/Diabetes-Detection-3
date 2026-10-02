# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.kts.

# PyTorch Mobile
-keep class org.pytorch.** { *; }
-keep class com.facebook.jni.** { *; }

# Keep model classes
-keep class com.diadet.madyapadma.model.** { *; }
-keep class com.diadet.madyapadma.ml.** { *; }

# CameraX
-keep class androidx.camera.** { *; }

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}

# Jetpack Compose
-keep class androidx.compose.** { *; }
