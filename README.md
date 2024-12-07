
# Iris Recognition and Template Security using Fuzzy Vault

## 🚀 Introduction

This repository contains the implementation of my **Minor Project** titled **"Iris Recognition and Template Security using Fuzzy Vault"**, completed under the guidance of **Prof. Ashok K. Bhateja** at **IIT Delhi**. The project combines **biometric recognition** and **cryptographic security** to develop a robust, secure iris authentication system leveraging the **fuzzy vault framework**.

The project addresses challenges such as template security, intra-user variations, and privacy preservation. Experimental results demonstrate its high recognition accuracy and secure template storage, making it suitable for modern biometric authentication systems.

---

## 📜 Features

- **Iris Recognition Pipeline**: Implements segmentation, normalization, and feature extraction for accurate biometric matching.
- **Template Security**: Ensures template revocability and privacy using **salting transformations** and **fuzzy vault construction**.
- **Robust Authentication**: Incorporates error correction techniques to handle intra-user variations and provide reliable recognition.
- **Privacy Protection**: Prevents reconstruction of the original biometric data even if the helper data is compromised.

---

## 🛠️ Project Workflow

### 1. **Iris Recognition Pipeline**
   - **Segmentation**: Isolates the iris region using methods like **Hough Transform** and **Daugman’s Integro-differential Operator**.
   - **Normalization**: Unwraps the iris region into a rectangular grid using the **Daugman’s Rubber Sheet Model**.
   - **Feature Extraction**: Extracts unique iris features using **Gabor Filters**, processed into a quantized feature vector for matching.

### 2. **Iris Template Security**
   - **Salting Transformation**:
     - Converts the ordered IrisCode into an unordered representation.
     - Encodes components with BCH codes to ensure error correction capability.
   - **Fuzzy Vault Construction**:
     - Secures the transformation key by embedding genuine points into a polynomial with chaff points for obfuscation.

### 3. **Authentication Process**
   - **Inverse Salting**: Decodes the fuzzy vault and reconstructs the original key using the query IrisCode.
   - **Vault Decoding**: Validates the decoded key against the stored data, ensuring secure access.


---

## 🧰 Technologies Used

- **Programming Language**: Python/Matlab (choose based on your implementation).
- **Image Processing**: Gabor Filters, Hough Transform.
- **Cryptography**: BCH Encoding/Decoding, Polynomial Interpolation.
- **Data**: IIT Delhi Iris Database, CASIA Iris Database.

---
