# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603409**
- **Họ và tên: PhanHaiDang**
- **SEED** (in ở ô đầu notebook): 2657

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.816      0.894        0.857   
2. BoW + MLP (deep learning)         0.989   0.893      0.939        0.919   
3a. Encoder (MiniLM) + kNN           0.933   0.951      0.942        0.950   
3b. Encoder (MiniLM) + LogReg        1.000   0.932      0.965        0.974   
3c. Encoder (MiniLM) + MLP head      0.970   0.932      0.950        0.974   
4a. LLM nhỏ zero-shot                1.000   0.600      0.750        0.750   
4b. LLM lớn zero-shot                0.909   1.000      0.952        0.952   
4c. LLM nhỏ few-shot                 0.870   1.000      0.930        0.930   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            2.963      46922.720       0.053   
2. BoW + MLP (deep learning)         4.642      44129.061       0.186   
3a. Encoder (MiniLM) + kNN         126.445         18.892     109.108   
3b. Encoder (MiniLM) + LogReg      128.848         47.827      27.598   
3c. Encoder (MiniLM) + MLP head    127.569         47.233      29.037   
4a. LLM nhỏ zero-shot                0.000          0.955     519.751   
4b. LLM lớn zero-shot                0.000          1.175     350.629   
4c. LLM nhỏ few-shot                 0.000          1.007     480.319   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.119         0.000  
2. BoW + MLP (deep learning)          0.320         0.000  
3a. Encoder (MiniLM) + kNN          205.367         0.000  
3b. Encoder (MiniLM) + LogReg        42.242         0.000  
3c. Encoder (MiniLM) + MLP head      37.080         0.000  
4a. LLM nhỏ zero-shot              2508.601         0.005  
4b. LLM lớn zero-shot              2446.511         0.010  
4c. LLM nhỏ few-shot               2607.521         0.013  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

  [07:16:07] emb_tricky: Encoder + LogReg
  [07:16:11] llm_models: Model nhỏ cho f1 cao hơn, model lớn đắt gấp 5 lần, llm có thắng mô hình embedding => ra embedding thắng
  [07:18:45] llm_fewshot: F1 tăng, chi phí mỗi tin tăng khoảng 2 lần => không tăng nhiều chi phí
Dự đoán sai nhiều ở phần chi phí

## 4. Phân tích 50 case khó

Model BoW tụt nhiều nhất, điểm trên tập test cao do các test ở trong môi trường lí tưởng hoá 2. Nhóm Spam giọng tự nhiên khó nhất với BoW, nhóm inject prompt khó nhất với llm, BoW đọc token rời rạc, không có = không vi phạm, LLm thì đọc token liền mạch và thực hiện hết các mệnh lệnh.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Chọn kết hợp BoW và LLM, BoW sẽ giải quyết những case dễ, LLM sẽ giải quyết những case khó hơn, chấp nhận đánh đổi độ trễ và chi phí cho những case khó hơn.

## 7. AI log và điều đã học

Hỏi cách implement function, nhận ra AI thường import thêm thư viện, cần yêu cầu sửa lại để dùng package có sẵn nếu có thể
