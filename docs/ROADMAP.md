# 🗺️ ROADMAP PHÁT TRIỂN HỆ THỐNG PHÁT HIỆN SỰ CỐ GIAO THÔNG

## 📋 TỔNG QUAN

Roadmap này mô tả kế hoạch phát triển hệ thống phát hiện sự cố giao thông theo 3 giai đoạn: **MVP → Hybrid → Production**, với các mục tiêu đo lường cụ thể và tiêu chí nghiệm thu rõ ràng.

---

## 🎯 PHASE 1: MVP (Minimum Viable Product)

### Mục tiêu đo lường

| Metric | Target | Mô tả |
|--------|--------|-------|
| **Recall** | ≥ 0.85 | Phát hiện được ít nhất 85% sự cố thực tế |
| **FAR (False Alarm Rate)** | ≤ 0.05 (5%) | Tối đa 5% cảnh báo sai |
| **MTTD (Mean Time To Detection)** | ≤ 10 giây | Phát hiện sự cố trong vòng 10 giây |
| **Latency p95** | ≤ 500ms | 95% requests xử lý trong 500ms |

### Task Breakdown (Tuần 1-4)

#### **Tuần 1: Baseline & Evaluation Protocol**
- [x] Hoàn thiện CNN baseline (MobileNetV2)
- [ ] Thiết kế evaluation protocol (train/val/test split, threshold tuning)
- [ ] Implement temporal confirmation module
- [ ] Tạo biểu đồ đánh giá (PR curve, ROC, confusion matrix)

#### **Tuần 2: Database & Storage**
- [ ] Thiết kế database schema (PostgreSQL)
- [ ] Implement SQLAlchemy models
- [ ] Tạo migration scripts
- [ ] Setup object storage cho media (S3/GCS hoặc local)

#### **Tuần 3: API & Serving**
- [ ] Hoàn thiện FastAPI endpoints
- [ ] Integrate temporal confirmation vào prediction pipeline
- [ ] Log predictions vào database
- [ ] Implement monitoring metrics (Prometheus)

#### **Tuần 4: Dashboard & Testing**
- [ ] Cập nhật Streamlit dashboard với temporal confirmation
- [ ] Thêm chức năng confirm/false_alarm incidents
- [ ] Unit tests cho các module mới
- [ ] Integration testing

### Rủi ro lớn nhất

**Rủi ro**: False alarm rate cao do model quá nhạy cảm
- **Cách giảm**: 
  - Implement temporal confirmation với K-frames và moving average
  - Tune threshold trên validation set theo mục tiêu FAR ≤ 5%
  - Thu thập thêm dữ liệu false positive để retrain

### Tiêu chí nghiệm thu (Definition of Done)

 **Hoàn thành Phase 1 khi:**
1. Model đạt Recall ≥ 0.85 và FAR ≤ 0.05 trên test set
2. Temporal confirmation module hoạt động và giảm FAR ít nhất 30%
3. Database schema đầy đủ, có migration scripts
4. API endpoints hoạt động, latency p95 ≤ 500ms
5. Dashboard hiển thị incidents và cho phép confirm/false_alarm
6. Có unit tests với coverage ≥ 70%
7. Tài liệu kỹ thuật đầy đủ (README, API docs)

---

##  PHASE 2: HYBRID (Vision + Sensor Fusion)

### Mục tiêu đo lường

| Metric | Target | Mô tả |
|--------|--------|-------|
| **Recall** | ≥ 0.90 | Phát hiện được ít nhất 90% sự cố |
| **FAR** | ≤ 0.03 (3%) | Giảm false alarm xuống 3% |
| **MTTD** | ≤ 8 giây | Phát hiện nhanh hơn |
| **Latency p95** | ≤ 300ms | Tối ưu latency |

### Task Breakdown (Tuần 5-8)

#### **Tuần 5: Sensor Data Integration**
- [ ] Thiết kế late fusion architecture
- [ ] Implement sensor data preprocessing
- [ ] Train baseline models cho sensor (Logistic Regression, XGBoost)
- [ ] So sánh Vision vs Sensor baselines

#### **Tuần 6: Hybrid Model**
- [ ] Implement late fusion (weighted average, voting)
- [ ] Train hybrid model trên combined dataset
- [ ] Evaluate hybrid vs single-modality models
- [ ] Tune fusion weights trên validation set

#### **Tuần 7: Real-time Pipeline**
- [ ] Integrate sensor data stream (Kafka hoặc REST API)
- [ ] Implement real-time fusion pipeline
- [ ] Optimize latency (batch processing, async)
- [ ] Load testing và performance tuning

#### **Tuần 8: Advanced Features**
- [ ] Implement early fusion (feature-level fusion)
- [ ] Add confidence calibration
- [ ] Create comparison dashboard (Vision/Sensor/Hybrid)
- [ ] Documentation và báo cáo

### Rủi ro lớn nhất

**Rủi ro**: Sensor data không sẵn có hoặc chất lượng kém
- **Cách giảm**: 
  - Sử dụng simulated sensor data từ video (vehicle counting, speed estimation)
  - Tạo synthetic sensor data từ annotations
  - Hybrid model có thể fallback về Vision-only nếu sensor data không có

### Tiêu chí nghiệm thu

 **Hoàn thành Phase 2 khi:**
1. Hybrid model đạt Recall ≥ 0.90 và FAR ≤ 0.03
2. Hybrid model tốt hơn Vision-only ít nhất 5% về F1-score
3. Real-time pipeline xử lý được ≥ 10 FPS
4. Latency p95 ≤ 300ms
5. Có so sánh công bằng Vision/Sensor/Hybrid baselines
6. Dashboard hiển thị predictions từ cả 3 modalities

---

## 🏭 PHASE 3: PRODUCTION

### Mục tiêu đo lường

| Metric | Target | Mô tả |
|--------|--------|-------|
| **Recall** | ≥ 0.95 | Phát hiện được 95% sự cố |
| **FAR** | ≤ 0.01 (1%) | False alarm rất thấp |
| **MTTD** | ≤ 5 giây | Phát hiện rất nhanh |
| **Latency p95** | ≤ 200ms | Latency tối ưu |
| **Uptime** | ≥ 99.9% | High availability |
| **Throughput** | ≥ 100 req/s | Xử lý nhiều requests |

### Task Breakdown (Tuần 9-12)

#### **Tuần 9: Production Infrastructure**
- [ ] Setup containerization (Docker, Kubernetes)
- [ ] Implement CI/CD pipeline
- [ ] Setup monitoring (Prometheus, Grafana)
- [ ] Configure auto-scaling

#### **Tuần 10: Model Optimization**
- [ ] Model quantization (INT8, FP16)
- [ ] Model pruning và distillation
- [ ] Optimize inference engine (TensorRT, ONNX)
- [ ] Benchmark performance

#### **Tuần 11: Advanced ML Features**
- [ ] Online learning / incremental training
- [ ] A/B testing framework
- [ ] Model versioning và rollback
- [ ] Automated retraining pipeline

#### **Tuần 12: Integration & Deployment**
- [ ] Integrate với external systems (traffic lights, alert system)
- [ ] Setup backup và disaster recovery
- [ ] Security hardening (authentication, encryption)
- [ ] Production deployment và smoke tests

### Rủi ro lớn nhất

**Rủi ro**: System không scale được hoặc downtime cao
- **Cách giảm**: 
  - Load testing sớm và thường xuyên
  - Implement circuit breakers và retry logic
  - Setup monitoring và alerting
  - Có backup plan (fallback models, manual review)

### Tiêu chí nghiệm thu

 **Hoàn thành Phase 3 khi:**
1. Tất cả metrics đạt target
2. System uptime ≥ 99.9% trong 1 tháng
3. Có CI/CD pipeline tự động
4. Monitoring và alerting hoạt động
5. Security audit passed
6. Documentation đầy đủ cho operations
7. Có runbook cho incident response

---

##  TỔNG KẾT METRICS THEO PHASE

| Metric | Phase 1 (MVP) | Phase 2 (Hybrid) | Phase 3 (Production) |
|--------|--------------|------------------|----------------------|
| Recall | ≥ 0.85 | ≥ 0.90 | ≥ 0.95 |
| FAR | ≤ 5% | ≤ 3% | ≤ 1% |
| MTTD | ≤ 10s | ≤ 8s | ≤ 5s |
| Latency p95 | ≤ 500ms | ≤ 300ms | ≤ 200ms |
| Throughput | - | - | ≥ 100 req/s |
| Uptime | - | - | ≥ 99.9% |

---

## 🔄 QUY TRÌNH ĐÁNH GIÁ VÀ ĐIỀU CHỈNH

1. **Weekly Review**: Đánh giá tiến độ mỗi tuần, điều chỉnh task nếu cần
2. **Phase Gate Review**: Trước khi chuyển phase, review tất cả tiêu chí nghiệm thu
3. **Retrospective**: Sau mỗi phase, rút kinh nghiệm và cập nhật roadmap

---

## 📝 GHI CHÚ

- **Dữ liệu**: Hiện tại có 46 ảnh (26 incident, 20 normal). Cần thu thập thêm để đạt target metrics.
- **Hạ tầng**: Local development hiện tại, cần chuẩn bị cho cloud deployment ở Phase 3.
- **Team**: Có thể cần thêm resources cho Phase 2-3 (ML engineer, DevOps).

---

*Cập nhật lần cuối: [Ngày hiện tại]*

