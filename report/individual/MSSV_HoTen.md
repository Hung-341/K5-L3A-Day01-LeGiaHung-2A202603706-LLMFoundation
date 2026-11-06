# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603706
- **Họ và tên:** Lê Gia Hưng
- **SEED** (in ở ô đầu notebook): 1432

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | F1 (test) | F1 (SUB 40) | p95 ms/tin | tin/s (cả lô) | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.888 | 0.947 | 0.08 | 52.954 | 0 |
| 2. BoW + MLP | 0.939 | 0.974 | 0.49 | 22.211 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.929 | 0.950 | 163.3 | 10.8 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.960 | 0.974 | 38.0 | 23.5 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.975 | 0.947 | 37.5 | 30.4 | 0 |
| 4a. LLM nhỏ zero-shot (Haiku) | — | 0.950 | 867.5 | 1.45 | 0.009 |
| 4b. LLM lớn zero-shot (Opus) | — | 1.000 | 2077.6 | 0.71 | 0.345 |
| 4c. LLM nhỏ few-shot (Haiku) | — | 0.947 | 1424.7 | 1.16 | 0.021 |

- Đường Pareto (1 − F1 trên SUB ↔ p95): **BoW + LogReg, BoW + MLP, Opus**.
- Đạt SLA p95 < 50 ms: **1, 2, 3b, 3c**.
- SUB chỉ 40 tin: 1 tin sai đổi F1 khoảng 2.5–2.7 điểm %, nên chênh < 0.05 coi như hoà.

## 3. Dự đoán và thực tế

1. **Stress test (`hard_eval`).** Tôi đoán LLM sẽ bị prompt injection lừa. Thực tế 4a và 4d đều đúng **100%** ở nhóm này, và LLM gần như không tụt (F1 0.950 trên SUB, 0.935–0.952 trên case khó). Lệch vì model Claude hiện đại đã được huấn luyện để không làm theo chỉ dẫn nằm trong dữ liệu. Ngược lại BoW tụt rất mạnh, từ 0.947 xuống 0.316.
2. **LLM lớn vs nhỏ (`llm_models`).** Tôi đoán F1 hai model gần nhau (chênh < 0.05). Lần chạy đầu Opus 1.000, Haiku 0.923 (chênh 0.077). Haiku sai các tin viết tắt, mơ hồ kiểu "RCT' THNQ Adrian for U text", cần kiến thức nền mà model nhỏ thiếu. Chạy lại thì Haiku lên 0.950, vì Claude không nhận `temperature=0` nên kết quả dao động giữa các lần chạy. Phần chi phí thì khớp: Opus đắt khoảng 40 lần.
3. **Tin nhắn ngụy trang (`emb_tricky`).** Tôi đoán Encoder + LogReg đúng nhiều hơn BoW. Thực tế hoà **4/6**. BoW sai tin "brand new phone at no cost" (không có từ khoá spam), nhưng lại đúng tin "complimentary holiday" mà encoder sai, vì cụm này có sẵn trong spam của bộ dữ liệu. Cả hai cùng sai tin phishing "account rewards expire tonight", vì kiểu spam này hiếm trong dữ liệu train.

## 4. Phân tích 50 case khó

Bảng dưới là lần chạy cuối với 55 case (50 case gốc + 5 case thêm vào `MY_CASES`).

| Model | F1 SUB | F1 case khó | Accuracy case khó |
| --- | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.947 | **0.316** | 0.53 |
| 2. BoW + MLP | 0.974 | 0.429 | 0.56 |
| 3a. Emb + kNN | 0.950 | 0.700 | 0.67 |
| 3b. Emb + LogReg | 0.974 | 0.549 | 0.58 |
| 3c. Emb + MLP head | 0.947 | 0.588 | 0.62 |
| 4a. LLM zero-shot | 0.950 | 0.935 | 0.93 |
| 4d. LLM + prompt phòng thủ | 0.950 | 0.952 | 0.95 |

- **BoW + LogReg tụt nhiều nhất** (0.947 xuống 0.316), sai nặng ở **spam giọng tự nhiên** (9%), **né từ khoá** (17%) và **thao túng** (11%). BoW chỉ cộng trọng số của các token rời rạc có trong từ điển, nên tin không có từ khoá thì không bắt được.
- **Embedding** yếu ở **thao túng** (3b 11%) và **nói về spam** (3a 20%), vì vector nghĩa của các tin này gần với lớp còn lại.
- **LLM** đọc gần như người nên chỉ sai lẻ tẻ.
- Test thường quá lạc quan vì test và train cùng nguồn, và các spam gần trùng (chỉ khác số điện thoại) vẫn nằm ở cả hai phía.

**Case cụ thể:**

- **#102** "Hey it's Tom from uni, changed my number. Are you around? Need a small favour, will explain" (spam): **mọi model đều sai**. Tin mạo danh không có link, từ khoá hay yêu cầu tiền. BoW không thấy từ spam nào, vector nghĩa gần tin trò chuyện, LLM cũng không có căn cứ để nghi ngờ. Chỉ đọc văn bản thì không phân biệt được, cần thêm thông tin người gửi.
- **#9** "Txt me ur address and I'll send the free tickets my boss gave me for the concert" (ham): **4d chặn nhầm**. Tin có "txt", "ur", "free" và xin địa chỉ, giống phishing. Prompt phòng thủ dặn model nghi ngờ mạnh nên đoán spam. Case #101 ("Send me your bank details… Free pizza") bị sai cùng kiểu.

**Prompt injection:** 4a và 4d đều đúng 100% ở nhóm này. Prompt phòng thủ nâng tổng từ 0.93 lên 0.95 nhưng làm chặn nhầm thêm tin ham có tín hiệu phishing. Đề xuất phòng thủ ngoài prompt: chuẩn hoá đầu vào và lọc câu lệnh bằng luật, đồng thời đặt model cổ điển làm lớp kiểm tra song song, không để LLM là lớp quyết định duy nhất.

## 5. Thử thách cá nhân

- **Đề #5:** Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.
- **Kết quả:** <!-- Ghi trung thực những gì bạn đã làm được trong 15 phút. Nếu chưa xong, ghi rõ "chưa hoàn thành" và phần đã làm. -->
- **Rút kinh nghiệm:** đề này gần giống Thí nghiệm 3.B (ô 58). Cách nhanh nhất là dùng lại code của ô đó, chỉ thay danh sách tin, rồi in trọng số các từ (`w[vocab[word]]`) để làm bằng chứng cho phần giải thích.

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (trung bình khoảng 116 tin/s), p95 < 50 ms/tin.

- **Đạt độ trễ:** BoW + LogReg (0.08 ms), BoW + MLP (0.49 ms), 3b (38 ms), 3c (37.5 ms). 3a và mọi LLM (868–2078 ms) đều vượt.
- **Chọn 3c** (F1 test 0.975, cao nhất trong nhóm đạt SLA, precision 1.000). Đánh đổi: chỉ khoảng 30 tin/s mỗi máy nên cần khoảng 4–5 máy hoặc GPU, và p95 37.5 ms còn ít khoảng dư so với 50 ms. Nếu ưu tiên chi phí hạ tầng, BoW + MLP (F1 0.939, 22.211 tin/s, một máy là đủ) là phương án dự phòng.
- **Chi phí nếu dùng LLM cho toàn bộ** (300 triệu tin/tháng): Haiku khoảng $2.700, Haiku few-shot khoảng $6.300, Opus khoảng $103.500. LLM cũng vượt SLA khoảng 17–40 lần, nên không dùng cho toàn bộ được.
- **Thiết kế router:** 3c xử lý mọi tin. Chỉ tin lưỡng lự (p ∈ [0.3, 0.7]) mới gửi lên LLM (Haiku + prompt phòng thủ, đúng 95% trên case khó). Nếu khoảng 5% số tin là lưỡng lự, chi phí LLM khoảng $135/tháng, và p95 vẫn dưới 50 ms vì dưới 5% số tin chịu độ trễ cao. Các model sai ở những chỗ khác nhau, nên kết hợp tốt hơn dùng một cách duy nhất.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1–3 (vùng tự code) | Hỏi khái niệm: data leakage, learning rate, overfitting, số tham số MLP, bias, cosine/chuẩn hoá, embedding vs BoW | Giải thích khái niệm, không viết code TODO (AI từ chối viết `top_k_similar`, chỉ giải thích các bước) | Tự viết code Phần 1–3, chạy các ô kiểm tra |
| 4 (vùng được dùng AI) | Hỏi cách làm và lỗi khi làm TODO 4.1–4.4; lỗi API key 401 | AI viết code `call_openai` (thêm client), `build_classification_prompt`, `parse_label`, `llm_classify_batch`; hướng dẫn dùng Colab Secrets, đổi provider | Đọc hiểu từng dòng trước khi chạy, chạy ô kiểm tra mock 4.1–4.4; đổi provider sang Claude; xoá output lỗi có lộ một phần key |
| Tự luận (Phần 0–5, phản ánh) | Thảo luận các câu hỏi, giải thích cơ chế, phân tích bảng kết quả của mình | Giải thích, gợi ý ý chính và bản trả lời ngắn | Tìm hiểu lại, viết câu trả lời bằng lời của mình, dùng số liệu của chính mình |
| Bonus MY_CASES | Gợi ý case khó | AI gợi ý 4 case, 1 case là của tôi | Chạy lại stress test, ghi nhận model sai #101, #102 |
| Tính tay | Hỏi cách tính | Hướng dẫn công thức và các bước | Tự tính trên giấy, kiểm tra hash |
| Thử thách cá nhân | Hỏi lời giải sau khi đã nộp | Lời giải mẫu để tự học | Không đưa vào bài nộp |

**Điều đã học:**

- Điểm cao trên tập test cùng nguồn không chứng minh model tốt ngoài đời: BoW đạt F1 0.947 trên SUB nhưng chỉ 0.316 trên case khó.
- Đổi **biểu diễn** (BoW sang embedding) giúp nhiều hơn thêm **độ sâu** (hidden layer): +0.072 F1 so với +0.015.
- Chọn model cho sản phẩm phải xét cả độ trễ, chi phí và số đo trên nhiều tập, không chỉ F1. Với tập nhỏ như SUB 40, chênh vài tin hay một lần chạy lại cũng đủ đổi thứ hạng.
