# pizza or steak or sushi判定アプリ
# modeltest.pyではconfusionマトリックスで性能チェック可能


## 概要
modelcustom2.pyでresnet18を転移学習させたモデルをmodel.pthとして保存します。
streamlit_app.pyでmodel.pthを読み込み、クライアントからアップロードされた画像を分類させます。

## 改善箇所
・基本的な転移学習をしただけでモデル単体の完成度は低く、作り込むにはepoch数を増やすだけではなく水増しなどの対策が必要　　



## 参考資料:
https://blog.streamlit.io/deep-learning-apps-for-image-processing-made-easy-a-step-by-step-guide/
(https://github.com/MainakRepositor/MLDP/assets/64016811/7287fa8f-e3b0-4db2-aa62-15f700671129)
https://qiita.com/kagami_t/items/60ee9d7f12ca0a593365
https://www.learnpytorch.io/03_pytorch_computer_vision/
