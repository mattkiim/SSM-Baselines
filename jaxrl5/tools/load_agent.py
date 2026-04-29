from typing import Any, Callable, Dict, Optional, Tuple

from jaxrl5.tools.load_cal import load_cal
from jaxrl5.tools.load_rac import load_rac
from jaxrl5.tools.load_sac_cbf import load_sac_cbf
from jaxrl5.tools.load_sac_lag import load_sac_lag
from jaxrl5.tools.load_ssm import load_ssm
from jaxrl5.tools.load_td3 import load_td3
from jaxrl5.tools.load_td3_lag import load_td3_lag


def load_agent(
    algo: str,
    ckpt_path: str,
    step: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[object, Callable, Dict]:
    algo = algo.strip().lower().replace("-", "_").replace(" ", "")
    if algo == "saclag":
        algo = "sac_lag"
    if algo == "td3":
        return load_td3(ckpt_path, step=step)
    if algo == "td3_lag":
        return load_td3_lag(ckpt_path, step=step)
    if algo == "cal":
        return load_cal(ckpt_path, step=step, **kwargs)
    if algo == "ssm":
        return load_ssm(ckpt_path, step=step, **kwargs)
    if algo == "rac":
        return load_rac(ckpt_path, step=step, **kwargs)
    if algo == "sac_lag":
        return load_sac_lag(ckpt_path, step=step, **kwargs)
    if algo == "sac_cbf":
        return load_sac_cbf(ckpt_path, step=step, **kwargs)
    raise ValueError(
        "Unsupported algo "
        f"'{algo}'. Expected one of: td3, td3_lag, cal, ssm, rac, sac_lag, sac_cbf"
    )
