# bcycle-jp: 日本景気循環モデル MVP

日本の景気循環を月次で5ステージ(回復/上昇/成熟/軟化/下降)に分類するモデル。
レベル(0-100) × モメンタム(z-score) のクアドラントで判定し、ヒステリシスで遷移の頑健性を確保。

---

## アーキテクチャ

```
indicators.yaml ─→ adapter (e-Stat/FRED/BBG) ─→ raw 系列
                                                    │
                                              transform (yoy/level)
                                                    │
                            rolling Z / percentile ─┴─ 各指標スコア
                                                    │
                                       重み付き合成 ──┴─ Cycle Level (0-100)
                                                                │
                                               3M変化のZ ─── Momentum
                                                                │
                                            quadrant + ヒステリシス ── Stage
```

**設計の3つの軸:**

1. **指標定義は yaml に外部化**(`config/indicators.yaml`)。各指標が複数ソースを持ち、実行時に切替可能
2. **アダプタパターン**で公的API/Bloombergをシームレスに切替(`DATA_SOURCE_PREFER` 環境変数)
3. **パラメータは settings.yaml に集約**。窓長・重み・閾値の sensitivity 分析が容易

---

## ディレクトリ構造

```
bcycle-jp/
├── README.md                       ← このファイル
├── pyproject.toml                  ← 依存定義
├── .env.example                    ← API キーのテンプレート
├── config/
│   ├── indicators.yaml             ← 10指標の定義(全ソース)
│   └── settings.yaml               ← サイクルパラメータ
├── src/bcycle_jp/
│   ├── adapters/
│   │   ├── base.py                 ✅ ABC 完成
│   │   ├── registry.py             ✅ ディスパッチャ完成
│   │   ├── estat.py                🔧 スケルトン(Claude Code 実装)
│   │   ├── fred.py                 🔧 スケルトン(Claude Code 実装)
│   │   └── bloomberg.py            🔧 スケルトン(Claude Code 実装)
│   └── core/
│       ├── loader.py               ✅ yaml→アダプタの統合層 完成
│       ├── normalize.py            ✅ 正規化ロジック 完成
│       ├── composite.py            ✅ レベル/モメンタム合成 完成
│       └── classify.py             ✅ ステージ判定+ヒステリシス 完成
├── scripts/
│   └── 01_jp_mvp.py                ✅ E2E 雛形(アダプタ実装後に走る)
├── tests/
│   └── test_normalize.py           ✅ 純ロジックのテスト 完成
└── data/                           ← parquet キャッシュ(gitignore)
```

✅ = 完成 / 🔧 = Claude Code への引き渡し対象

---

## セットアップ(PC作業)

```bash
# クローン後、仮想環境を作る(uv推奨、pipでも可)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 環境変数
cp .env.example .env
# .env を編集して ESTAT_APP_ID、FRED_API_KEY を入れる

# テスト(純ロジック層)
pytest tests/

# E2E(Claude Code でアダプタ実装後)
python scripts/01_jp_mvp.py
```

API キー取得:
- e-Stat: https://www.e-stat.go.jp/api/api-info/api-guide
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html

---

## 🚦 Claude Code への引き渡しポイント

### Phase 1(完成済み — 引き渡し不要)

純ロジック層は外部依存ゼロで動く。以下は実装+テスト済み:

- `core/normalize.py` — YoY化、ローリング Z、ローリングパーセンタイル、Winsorize
- `core/composite.py` — `compute_level()` と `compute_momentum()`
- `core/classify.py` — 5ステージのクアドラント判定 + ヒステリシス + 確度計算
- `core/loader.py` — yaml と adapter を繋ぐ統合層
- `adapters/base.py`, `adapters/registry.py` — インターフェース
- `tests/test_normalize.py` — pytest 実行可能

### Phase 2(Claude Code に渡す作業)

優先順:

**① e-Stat アダプタ実装** (`adapters/estat.py`)

   - `getStatsData` で時系列取得
   - JSON の `DATA_INF.VALUE` をパースして `pd.Series` に
   - `stats_data_id == "TBD"` の指標を `stats_code` から `getStatsList` で動的解決
   - **同時に `config/indicators.yaml` の `TBD` を実IDで埋める**
   - `data/{indicator_id}.parquet` にキャッシュ

**② FRED アダプタ実装** (`adapters/fred.py`)

   - `fredapi` を使う(`uv pip install fredapi` 済)
   - クロスチェック用途。実装は10行程度で済むはず
   - yaml の FRED 用 `series_id` の `# 要確認` を実IDに置換

**③ 派生指標の formula 評価** (`core/loader.py` または `scripts/01_jp_mvp.py`)

   - `real_policy_rate = tona_rate - core_cpi_yoy` を実際に計算
   - TONA レートは日銀統計から(`adapters/boj.py` を追加するのが綺麗)
   - JGB 10Y/2Y のスプレッドも同様

**④ E2E スクリプトを実行 → 出力検証**

   - `python scripts/01_jp_mvp.py` でレベル/モメンタム/ステージが出る
   - **内閣府の景気基準日付**と並べてプロット
   - 1991/1997/2001/2008/2020 の5回のリセッションを捕捉できているか
   - 漏れがあれば指標差し替えか重み調整

**⑤ Jupyter notebook 作成** (`notebooks/01_jp_mvp.ipynb`)

   - スクリプト相当を notebook 化
   - ステージ分布、各指標の z-score 推移、クアドラント散布図を可視化

### Phase 3(MVP 後の拡張)

- セクター/ファクター・マッピング(過去ステージ別の超過リターン検証)
- 米国・韓国・台湾への展開(adapter を追加するだけで yaml の差替えで対応)
- ステージ判定の信頼区間(ブートストラップ or ベイズ)
- 動的因子モデル(Stock-Watson)への発展

---

## 主要な設計判断(ここで議論済み)

1. **5ステージはレベル × モメンタムのクアドラント離散化**(理論的必然性は無く、Investment Clock との折衷)
2. **ローリング窓は10年**(min 5年)。レジーム変化に追従しつつ短期ノイズを排除
3. **MVP は percentile_mean 合成**。分布の裾に頑健。z_mean は v2 で選択可能に
4. **ヒステリシスは2期**。フリップフロップ防止、感度はパラメータ化
5. **指標は10本月次に固定**。短観(四半期)は v2 でエンリッチメント

---

## ライセンス・補足

- 内部リサーチ用途を想定
- 公開する場合は各データソースの利用規約を確認(e-Stat は出典明示で再配布可)
