"""只评测、不训练：从已保存的最优 checkpoint 直接算出 valid / test 指标。

为什么单独写这个脚本
--------------------
PyTorch 2.6 起，`torch.load` 的 `weights_only` 默认值从 `False` 变成了 `True`。
RecBole 1.2.1 的 `trainer/trainer.py:583` 用的是裸 `torch.load(...)`：

    checkpoint = torch.load(checkpoint_file, map_location=self.device)

所以「训练结束后加载最优模型做 test 评测」这一步会抛
`_pickle.UnpicklingError: Weights only load failed`。
训练本身没事（它只 save 不 load），但 test 指标就此丢失。

这个脚本在 `import recbole` 之前把默认值改回 `False`（checkpoint 是本项目自己产出的，
可信），然后复用 `load_data_and_model` 把模型和数据集一起加载，只跑评测。

用法
----
    cd ~/genrec
    python scripts/eval_only.py --model_file saved/SASRec-Sep-28-2026_19-18-04.pth
"""
import argparse
import functools
from logging import getLogger

import torch

# ---- 必须在 recbole 之前打补丁 ----
_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_weights_only_false
# ----------------------------------

from recbole.quick_start import load_data_and_model  # noqa: E402
from recbole.trainer import Trainer  # noqa: E402
from recbole.utils import dict2str  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_file", required=True, help="saved/ 下的 .pth 路径")
    args = ap.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=args.model_file
    )

    # 坑：load_data_and_model 是从 checkpoint 里反序列化出 Config 对象的，
    # Config.__init__ 没跑过 → compatibility_settings() 也没跑过 →
    # numpy 的旧别名（np.float / np.bool / np.object ...）没被补回来。
    # 而 evaluator/metrics.py:190 的 NDCG 正要用 np.float。手动补一次。
    config.compatibility_settings()

    logger = getLogger()
    logger.info("=" * 64)
    logger.info("EVAL ONLY (no training). checkpoint = %s", args.model_file)
    logger.info("=" * 64)

    trainer = Trainer(config, model)

    valid_result = trainer.evaluate(valid_data, model_file=args.model_file)
    logger.info("valid result:\n%s", dict2str(valid_result))

    test_result = trainer.evaluate(test_data, model_file=args.model_file)
    logger.info("test result:\n%s", dict2str(test_result))

    # 同时打到 stdout，方便 `| tee` 直接抓
    print("\n===== EVAL ONLY RESULT =====")
    print("checkpoint:", args.model_file)
    print("valid:", dict2str(valid_result))
    print("test :", dict2str(test_result))


if __name__ == "__main__":
    main()
