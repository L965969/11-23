# ===========================================================================
# ANN 課後作業（二）｜糖尿病病程進展預測（Diabetes Progression）
# 研究任務：以 Linear Regression 與單隱藏層 ANN 預測一年後病程進展指標 (progression)
# 資料來源：scikit-learn load_diabetes(scaled=False)（n = 442, p = 10）
# 注意：本教材僅供統計與機器學習教學，不可用於臨床診斷或醫療決策。
# ===========================================================================

# [套件與工作目錄設定]
if (!requireNamespace("nnet", quietly = TRUE)) {
  stop("尚未安裝 nnet 套件，請先執行 install.packages('nnet')。")
}
set.seed(20261025)

target_dir <- "D://"
if (dir.exists(target_dir)) {
  setwd(target_dir)
}

data_file <- "diabetes_progression.csv"
if (!file.exists(data_file)) {
  stop("找不到 diabetes_progression.csv，請確認工作目錄是否正確。")
}

dat <- read.csv(data_file, stringsAsFactors = FALSE, check.names = FALSE)

# 定義 10 個解釋變數與連續型目標變數 progression
features <- c("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")
target   <- "progression"

# 確認欄位存在且無缺失值
stopifnot(all(c(features, target) %in% names(dat)))
stopifnot(!anyNA(dat[, c(features, target)]))

out_dir <- "Results_糖尿病病程進展"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ===========================================================================
# [TODO 1] 探索性資料分析 (EDA)：Y 分布與 BMI 散點圖
# ===========================================================================
cat("[STEP 1] 資料檢查與摘要統計\n")
cat("樣本數：", nrow(dat), "；解釋變數數：", length(features), "\n")
print(summary(dat[, c(features, target)]))

png(file.path(out_dir, "01_EDA_distribution_and_bmi.png"), width = 1200, height = 550, res = 145)
par(mfrow = c(1, 2))
hist(dat$progression, breaks = 20, col = "#3B7A8C", border = "white",
     xlab = "One-year progression index", ylab = "Count",
     main = "Disease progression index")
plot(dat$bmi, dat$progression, pch = 19, col = rgb(0.75, 0.42, 0.32, 0.65),
     xlab = "BMI (raw dataset scale)", ylab = "Progression Index",
     main = "BMI versus progression")
abline(lm(progression ~ bmi, data = dat), col = "#2D4B5A", lwd = 2, lty = 2)
invisible(dev.off())

# ===========================================================================
# [資料切割] Train (60%) / Validation (20%) / Test (20%)
# ===========================================================================
n <- nrow(dat)
shuffled_idx <- sample(seq_len(n))

n_train <- floor(0.60 * n)
n_valid <- floor(0.20 * n)

train_idx <- shuffled_idx[1:n_train]
valid_idx <- shuffled_idx[(n_train + 1):(n_train + n_valid)]
test_idx  <- shuffled_idx[(n_train + n_valid + 1):n]

train_raw <- dat[train_idx, c(features, target), drop = FALSE]
valid_raw <- dat[valid_idx, c(features, target), drop = FALSE]
test_raw  <- dat[test_idx,  c(features, target), drop = FALSE]

cat(sprintf("\n[STEP 2] 資料切割完成 -> Train: %d, Validation: %d, Test: %d\n",
            nrow(train_raw), nrow(valid_raw), nrow(test_raw)))

# ===========================================================================
# [TODO 2] 僅由訓練集估計 X 與 Y 的縮放參數（防止資料洩漏）
# ===========================================================================
fit_scaler <- function(df, x_cols, y_col) {
  x_mean <- vapply(df[x_cols], mean, numeric(1))
  x_sd   <- vapply(df[x_cols], sd,   numeric(1))
  x_sd[x_sd == 0] <- 1
  y_mean <- mean(df[[y_col]])
  y_sd   <- sd(df[[y_col]])
  if (y_sd == 0) y_sd <- 1
  list(x_mean = x_mean, x_sd = x_sd, y_mean = y_mean, y_sd = y_sd)
}

transform_data <- function(df, scaler, x_cols, y_col) {
  z_x <- sweep(as.matrix(df[, x_cols, drop = FALSE]), 2, scaler$x_mean, "-")
  z_x <- sweep(z_x, 2, scaler$x_sd, "/")
  out <- as.data.frame(z_x)
  out$y_std <- (df[[y_col]] - scaler$y_mean) / scaler$y_sd
  out$y_raw <- df[[y_col]]
  out
}

unscale_y <- function(y_star, scaler) {
  as.numeric(y_star) * scaler$y_sd + scaler$y_mean
}

# 評估指標函數：RMSE, MAE, R2 (使用評估集自身 y 的平均數計算 SST)
reg_metrics <- function(y_true, y_pred) {
  sse <- sum((y_true - y_pred)^2)
  sst <- sum((y_true - mean(y_true))^2)
  c(
    RMSE = sqrt(mean((y_true - y_pred)^2)),
    MAE  = mean(abs(y_true - y_pred)),
    R2   = 1 - sse / sst
  )
}

scaler_tr <- fit_scaler(train_raw, features, target)
train_sc  <- transform_data(train_raw, scaler_tr, features, target)
valid_sc  <- transform_data(valid_raw, scaler_tr, features, target)

# ===========================================================================
# [TODO 3] 基準 Linear Regression 與三種架構 ANN (size = 3, 6, 10) 的 Validation 比較
# ===========================================================================
cat("\n[STEP 3] 在 Validation 集比較 Linear Regression 與 ANN-3 / ANN-6 / ANN-10\n")

# 1. Linear Regression (在 Train 擬合，在 Validation 評估)
lm_val_fit  <- lm(progression ~ ., data = train_raw)
pred_lm_val <- as.numeric(predict(lm_val_fit, newdata = valid_raw))
m_lm_val    <- reg_metrics(valid_raw$progression, pred_lm_val)

# 2. ANN 候選架構 (size = 3, 6, 10)，使用標準化後的 X 與 Y，linout = TRUE
candidate_sizes <- c(3, 6, 10)
ann_val_results <- lapply(candidate_sizes, function(s) {
  set.seed(20261025)
  fit <- nnet::nnet(
    y_std ~ .,
    data   = train_sc[, c(features, "y_std")],
    size   = s,
    linout = TRUE,
    decay  = 0.01,
    maxit  = 900,
    trace  = FALSE
  )
  pred_std <- predict(fit, newdata = valid_sc[, features, drop = FALSE])
  pred_raw <- unscale_y(pred_std, scaler_tr)
  metrics  <- reg_metrics(valid_raw$progression, pred_raw)
  data.frame(
    Model       = paste0("ANN-", s),
    HiddenUnits = s,
    Val_RMSE    = metrics["RMSE"],
    Val_MAE     = metrics["MAE"],
    Val_R2      = metrics["R2"],
    row.names   = NULL
  )
})

val_table <- rbind(
  data.frame(Model = "Linear Regression", HiddenUnits = NA,
             Val_RMSE = m_lm_val["RMSE"], Val_MAE = m_lm_val["MAE"],
             Val_R2 = m_lm_val["R2"], row.names = NULL),
  do.call(rbind, ann_val_results)
)
print(val_table)
write.csv(val_table, file.path(out_dir, "02_validation_selection.csv"), row.names = FALSE)

# 僅根據 Validation RMSE 選出最佳 ANN 隱藏層神經元數
ann_only <- val_table[!is.na(val_table$HiddenUnits), ]
best_size <- ann_only$HiddenUnits[which.min(ann_only$Val_RMSE)]
cat(sprintf("-> 依 Validation RMSE 選定最佳 ANN 架構：size = %d\n", best_size))

# ===========================================================================
# [TODO 4-5] 合併 Train + Validation 重新估計最終模型，並於封存 Test 集最終評估
# ===========================================================================
cat("\n[STEP 4] 合併 Train + Validation 重新訓練，並評估 Test 集\n")
trainval_raw <- rbind(train_raw, valid_raw)

# 僅使用 Train + Validation 重新計算縮放參數（絕不碰觸 Test 資料）
scaler_tv   <- fit_scaler(trainval_raw, features, target)
trainval_sc <- transform_data(trainval_raw, scaler_tv, features, target)
test_sc     <- transform_data(test_raw,     scaler_tv, features, target)

# 最終 Linear Regression
final_lm <- lm(progression ~ ., data = trainval_raw)
pred_lm_test <- as.numeric(predict(final_lm, newdata = test_raw))

# 最終 ANN (使用選定的 best_size)
set.seed(20261025)
final_ann <- nnet::nnet(
  y_std ~ .,
  data   = trainval_sc[, c(features, "y_std")],
  size   = best_size,
  linout = TRUE,
  decay  = 0.01,
  maxit  = 900,
  trace  = FALSE
)
pred_ann_test_std <- predict(final_ann, newdata = test_sc[, features, drop = FALSE])
pred_ann_test     <- unscale_y(pred_ann_test_std, scaler_tv)

# ===========================================================================
# [TODO 6-7] Test 指標比較表、預測結果 CSV 與兩種診斷圖
# ===========================================================================
y_test <- test_raw$progression
test_report <- rbind(
  Linear_Regression = reg_metrics(y_test, pred_lm_test),
  ANN_Final         = reg_metrics(y_test, pred_ann_test)
)
rownames(test_report)[2] <- paste0("ANN (size=", best_size, ")")
print(round(test_report, 4))

write.csv(data.frame(Model = rownames(test_report), test_report, row.names = NULL),
          file.path(out_dir, "03_test_metrics.csv"), row.names = FALSE)

pred_out <- data.frame(
  Actual_Y      = y_test,
  Pred_LM       = pred_lm_test,
  Residual_LM   = y_test - pred_lm_test,
  Pred_ANN      = pred_ann_test,
  Residual_ANN  = y_test - pred_ann_test
)
write.csv(pred_out, file.path(out_dir, "04_test_predictions.csv"), row.names = FALSE)

# 圖 1：Observed vs Predicted (一致座標範圍 + y = x 參考線)
lim_obs <- range(c(y_test, pred_lm_test, pred_ann_test))
png(file.path(out_dir, "05_observed_vs_predicted.png"), width = 1200, height = 580, res = 145)
par(mfrow = c(1, 2))
plot(y_test, pred_lm_test, pch = 19, col = "#2D7185",
     xlim = lim_obs, ylim = lim_obs,
     xlab = "Actual Y (progression)", ylab = "Predicted Y",
     main = "Observed vs Predicted: Linear Regression")
abline(a = 0, b = 1, lty = 2, lwd = 2, col = "gray40")

plot(y_test, pred_ann_test, pch = 19, col = "#D06F64",
     xlim = lim_obs, ylim = lim_obs,
     xlab = "Actual Y (progression)", ylab = "Predicted Y",
     main = sprintf("Observed vs Predicted: ANN (size=%d)", best_size))
abline(a = 0, b = 1, lty = 2, lwd = 2, col = "gray40")
invisible(dev.off())

# 圖 2：Residual vs Fitted (一致座標範圍 + 殘差 = 0 水平線)
res_lm  <- y_test - pred_lm_test
res_ann <- y_test - pred_ann_test
xlim_fit <- range(c(pred_lm_test, pred_ann_test))
ylim_res <- range(c(res_lm, res_ann))

png(file.path(out_dir, "06_residual_vs_fitted.png"), width = 1200, height = 580, res = 145)
par(mfrow = c(1, 2))
plot(pred_lm_test, res_lm, pch = 19, col = "#2D7185",
     xlim = xlim_fit, ylim = ylim_res,
     xlab = "Fitted / Predicted Y", ylab = "Residual (Y - Y_hat)",
     main = "Residual vs Fitted: Linear Regression")
abline(h = 0, lty = 2, lwd = 2, col = "gray40")

plot(pred_ann_test, res_ann, pch = 19, col = "#D06F64",
     xlim = xlim_fit, ylim = ylim_res,
     xlab = "Fitted / Predicted Y", ylab = "Residual (Y - Y_hat)",
     main = sprintf("Residual vs Fitted: ANN (size=%d)", best_size))
abline(h = 0, lty = 2, lwd = 2, col = "gray40")
invisible(dev.off())

cat("\n完成！所有結果表格與三張 PNG 圖檔已儲存於：", normalizePath(out_dir), "\n")