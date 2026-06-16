---
layout: post
title:  "豊橋技術科学大学の新しいクラスタ計算機を試す"
categories: blog
tags:  blog
author: けーさん/こまたん
---

つい最近，私が所属している豊橋技術科学大学に新しいクラスタ計算機が導入されたので，それを使ってみた感想を書こうと思います．
特に，以前のクラスタでは自作のSingularityイメージを動かすことはできなかったのですが，新しいシステムでは自作のSingularityイメージを動かすことができるようですので，それをメインで書きたいと思います．

<!--more-->


## ローカルにSingularityを入れる

Singularity用のイメージを自作するために，Singularityをローカル環境に導入します．
とはいえ，SingularityのDockerイメージ[quay.io/singularity/singularity](https://quay.io/repository/singularity/singularity)が公開されていますので，これを使います．

```sh
$ docker pull quay.io/singularity/singularity
```

このSingularityのDockerイメージを使って，以下のように普段使っているDockerイメージをSingularityイメージへ変換できます．

```sh
# ローカル環境で使用しているDockerイメージをディスク上に出力
$ docker save [イメージ名] -o [出力先].tar
# 出力したイメージからSingularityイメージファイル（.sif）をビルド
$ docker run --rm   -v $(pwd):/work   -w /work   --privileged   quay.io/singularity/singularity:v4.3.1 build [変換後のsifファイル名].sif docker-archive://[ディスク上に出力したDockerイメージのファイル].tar
# 出てきた.sifファイルの所有権をrootから自ユーザーへ変更
$ sudo chown $(id -u):$(id -g) [変換後のsifファイル名].sif
# 不要なファイルを消しておく
$ rm [ディスク上に出力したDockerイメージのファイル].tar
```

## Singularityで自作のイメージを動かしてみる（開発ノード）

開発ノードでSingularityイメージを動かしてみます．

```sh
$ singularity exec /work/{ユーザー名}/[変換後のsifファイルのパス].sif /bin/bash
```

シェルが出てこればOKです．


## Singularityで自作のイメージを動かしてみる（計算ノード）

インタラクティブジョブで自分で作ったSingularityイメージを動かしてみます．
まずは，クラスタ計算機の`/work/{ユーザー名}`以下に適当に先ほど作った`.sif`ファイルを保存しておきます．
そして，以下のコマンドでインタラクティブジョブを投入します．

```sh
$ qsub -I -q Eduq -l select=1:ncpus=2:mem=16gb -v SINGULARITY_IMAGE=/work/{ユーザー名}/[変換後のsifファイルのパス].sif /bin/bash
```

少し待って，シェルが出てきたらOKです．


## Singularityコンテナの中からジョブスケジューラを利用する

当初はSingularityコンテナ内からジョブの投入など，ジョブスケジューラに対して接続して何かすることができなかったのですが，少し前にIMCの方で対応していただきました．
ありがとうございます．
大変助かります．

やり方は[wikiに記載の通り](https://hpcportal.imc.tut.ac.jp/wiki/ClusterSystemTips#singularity.2BMLMw8zDGMMpRhTBLMIkwuDDnMNYwuTCxMLgw5TD8MOkwklIpdSg-)です．


## AMD GPUでOpenDPDを動かす

AMD GPU（MI210）を用いて[OpenDPD](https://github.com/lab-emi/OpenDPD)を動かせたのでそのメモも以下にまとめておきます．
基本的にはpython3.12でROCm対応のTorchを入れて，`HIP_VISIBLE_DEVICES=0`を指定するだけでした．

まずはTorch環境を作るためにノードにログインします．

```sh
# ログインシェルをbashに指定（-S /bin/bash）してインタラクティブジョブを投入
$ qsub -I -q Eduq -l select=1:ncpus=1:mem=8gb:vnode=ysnd00:ngpus=1 -v SINGULARITY_IMAGE=/common/Singularity_sif/prg_env_amd_2025.02.sif -S /bin/bash
```

そして，OpenDPDのREADMEに記載の通りにcondaで環境を作ります．
ただし，バージョンを3.12にします．
（3.13だとうまくいきませんでしたので，geminiと相談しながらやっていたら3.12で成功しました）

```sh
$ conda create -n opendpd python=3.12 numpy scipy pandas matplotlib tqdm rich
$ conda activate opendpd
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

最後に，正しくGPUが認識されているか確認します．

```sh
$ python3 -c "import torch; print('バージョン:', torch.__version__); print('HIP(ROCm)認識:', torch.version.hip)"
```

今回は出力は以下のようになりました．
バージョン番号末尾に`+rocm6.2`とついてるのでうまく行っている感じです．

```
バージョン: 2.5.1+rocm6.2
HIP(ROCm)認識: 6.2.41133-dd7f95766
```

最後に，GPUを使う際には`HIP_VISIBLE_DEVICES=0`のように使用するGPUの指定が必要です．
これを`...=1`にすれば2枚目のGPUを使うし，`=0,1`にすれば2枚使うし，のようです．
とりあえず1枚しか使わない（OpenDPDは1枚しか使えない？）し，PBSやSingularityがうまくやってくれているようなので，`=0`固定でOKなようです．
なので，OpenDPDのPAモデル学習は以下のように実行できます．

```sh
$ HIP_VISIBLE_DEVICES=0 ACCELERATOR=cuda bash bash_scripts/train_all_pa.sh
```

