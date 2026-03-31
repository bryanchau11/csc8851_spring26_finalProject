import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from app.ui import build_demo  # type: ignore
from app.pipeline import predict, update_ingredient_portions  # type: ignore
from app.core_models import pipeline_mode, food_clf, _gpu_stats_md  # type: ignore


demo = build_demo(
    predict=predict,
    update_ingredient_portions=update_ingredient_portions,
    gpu_stats_md=_gpu_stats_md,
    pipeline_mode=pipeline_mode,
    food_clf=food_clf,
    share=True,
)


if __name__ == '__main__':
    demo.launch(share=True)
