# 🚘 ANPR (Automatic Number Plate Recognition)

This project implements an **Automatic Number Plate Recognition (ANPR)** system using artificial intelligence models and image processing techniques. The system detects the license plate, rotates it to a horizontal position, segments it, and recognizes its characters, enabling efficient automation of vehicle identification processes.


## 📖 Description  
The system detects license plates in images and extracts the license plate text using two models pre-trained with **YOLOv8**:
- One to locate the **license plate*** with 99% accuracy.
- Another to recognize the **characters** with 69% accuracy.


## ⚙️ Dependencies 

- [🔗 Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO)  

- [🛠️ Supervision](https://github.com/roboflow/supervision)  


👉 Recommended installation:  
```bash
pip install ultralytics supervision polars
```

---

## ▶️ Usage  
1. Run `main.py`  
2. Enter the test image number (**1 – 30**) when prompted.  
3. The system will:  
   - 🖼️ Displays the image of the detected plate
   - 🔤 The image shows the number of the detected license plate.  


## 🔄 Main Code Flow  
1. 📥 Load YOLO models  
2. 🖼️ Read selected image  
3. 🎯 Detect plate and extract coordinates  
4. ✂️ Crop & rotate plate image  
5. 🤖 Apply OCR model for character recognition  
6. 👁️ Display results with plate text overlay  


## 📷 Example

<img src="example.png" alt="Ejemplo" width="300"/>

## 📬 Contact

For questions or feedback, contact at `billysoplin@gmail.com`

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---