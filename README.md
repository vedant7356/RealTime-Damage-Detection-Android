# 📦 Real-Time Damage Detection - Android App

An Android-based real-time product damage detection system using on-device machine learning.  
Built for warehouse automation, inventory quality checks, and logistics where quick detection of product damage is essential.

---

## 🎯 Problem Statement

Manual inspection of products or packages in warehouses is slow, error-prone, and not scalable. Minor damages like dents, scratches, or packaging defects are often missed. This app allows real-time automated damage detection using a simple mobile device.

---

## 🚀 Solution Overview

This prototype app uses:

- 📱 Live Camera Feed (Camera2 API)
- 🧠 TensorFlow Lite (Quantized int8 model)
- 📊 Custom-trained ML model for product damage classification

The system instantly analyzes live camera frames and classifies objects into one of these categories:

- ✅ **No Damage**
- 🔧 **Scratch**
- 🔨 **Dent**

Warehouse staff can quickly sort products into safe or damaged zones for further processing.

---

## 🔧 Technologies Used

| Technology | Description |
|-------------|-------------|
| Android Studio | Native Android app (Kotlin) |
| Camera2 API | Live camera feed handling |
| TensorFlow Lite | Lightweight on-device ML inference |
| Teachable Machine | Used for model training and export |
| Kotlin + XML Layouts | Full app development stack |
| GitHub | Version control and code hosting |

---

## 📊 Dataset Details

We simulated a real warehouse scenario using a simple object: **an empty Sting bottle**  
We collected custom data for 3 core damage classes:

- No Damage (perfect condition)
- Scratch (surface-level scratches)
- Dent (structural deformation)

Dataset captured with:
- iPhone camera
- Multiple angles (front, side, top-down, tilted)
- Lighting variations (natural & artificial)
- Both full object views and close-up shots

---

## 🏗 App Architecture

1️⃣ Capture live camera frames using Camera2 API  
2️⃣ Preprocess frame (resize, normalize, convert to quantized input tensor)  
3️⃣ Run TensorFlow Lite model inference on-device  
4️⃣ Display real-time prediction probabilities on-screen

---

## 🔐 Model Architecture

- TensorFlow Lite Quantized (int8) Classification Model
- Input Size: 224 x 224 x 3  
- Output: 3-Class probabilities  
- Fully trained offline using **Teachable Machine**

---

## 🎯 Use Case Impact

- ✅ Fast warehouse inspection
- ✅ Automated sorting between damaged vs safe items
- ✅ Offline-capable — no cloud dependency
- ✅ Edge-ready solution — deployable to low-cost devices

---

## 📸 Demo Screenshots
![Non Damage Detection](https://github.com/user-attachments/assets/986ecc8c-2628-4308-ac46-7470027014e9) ![Product with scratches](https://github.com/user-attachments/assets/4eadd145-6c22-4b47-82f0-604e4cb60e9e) ![Product with dent](https://github.com/user-attachments/assets/b98ecc0e-7413-4322-8242-d5e285e43465) 



## 🚩 Future Scope

- Expand dataset with larger real-world variety
- Add additional damage classes (Torn Label, Crushed Corner, Broken Seal, etc.)
- Integrate object detection models for bounding-box level damage localization
- Full production-ready edge AI solution for warehouses and logistics companies

---

## 🙌 Credits

- Designed, developed & trained by **Vedant Prabhu**  
- Built for educational & prototype warehouse automation demo.

---

## 📄 License

This project is provided for prototype & demonstration purposes.

---

