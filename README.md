# Mango Leaf Disease Prediction
マンゴーの葉の写真をアップロードすると７つの病気か健康な状態の８クラスに分類するWebアプリです。
使用した言語はpythonのみで、フロントエンドはStreamlitを使用しています。
ローカル環境で試すにはDockerfile(準備中)を元にコンテナを作成するか、すでにこのデータ用に転移学習させたmodel.pthとstreamlit_app.pyをダウンロードすれば使用可能です。
＊サンプルのmodel.pthは1epochしか学習していません。
＊トレーニングに使用していない健康な葉のサンプル写真としてJPEGをリポジトリ内においてあります。
*URLからアクセス:https://stmangoapp-jjyomexbto2kjvglciuoue.streamlit.app/

## 概要
modelcustom.pyでresnet18を転移学習させたモデルをmodel.pthとして保存します。この際に学習に使用するデータはarchiveの名前で同じディレクトリ内にあると想定しています。
streamlit_app.pyでmodel.pthを読み込み、クライアントからアップロードされた画像を分類させます。

## 改善箇所
・トレーニングデータにない病気に対する分類ができない　　

・基本的な転移学習をしただけでモデル単体の完成度は低く、作り込むにはepoch数を増やすだけではなく水増しなどの対策が必要　　

・病気の葉のデータを個人が探すことは難しい

## Training用　Dataset Link:
https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset

## 参考資料:
https://blog.streamlit.io/deep-learning-apps-for-image-processing-made-easy-a-step-by-step-guide/
(https://github.com/MainakRepositor/MLDP/assets/64016811/7287fa8f-e3b0-4db2-aa62-15f700671129)
https://qiita.com/kagami_t/items/60ee9d7f12ca0a593365
