# Gradience: Automated LC-MS Gradient Optimization

Gradience is a Python package for automated HPLC gradient optimization in LC-MS workflows. By combining forward synthesis prediction, retention time modeling, and Bayesian optimization, it designs optimal gradients that maximize chromatographic separation of reaction products.

The system predicts likely products from a reaction description, estimates their retention times under candidate gradients, and efficiently searches the gradient parameter space using Trust Region Bayesian Optimization (TuRBO). The result is an optimized HPLC method tailored to your specific reaction mixture, reducing manual method development time and improving analytical separation.

Current Version: **1.0.0**

## 🔧 Installation & Usage

The conda environment detailed in `gradenv.yml` is required for Gradience to function properly.

Python 3.10+ is required.

```bash
conda env create -f gradenv.yml
conda activate gradenv
playwright install chromium
```

Check out `gradience_demo.ipynb` for a complete working example.


## 🧬 About Gradience

### Product Prediction

Product prediction uses [ASKCOS](https://askcos.mit.edu). A headless web scraper with Playwright navigates to the ASKCOS forward synthesis page, enters reactants and solvent, then downloads predicted products with probability scores and molecular weights.

```
@article{askcos2025,
  title={ASKCOS: Open-Source, Data-Driven Synthesis Planning},
  author={Zhengkai Tu and Sourabh J. Choure and Mun Hong Fong and Jihye Roh and Itai Levin and Kevin Yu and Joonyoung F. Joung and Nathan Morgan and Shih-Cheng Li and Xiaoqi Sun and Huiqian Lin and Mark Murnin and Jordan P. Liles and Thomas J. Struble and Michael E. Fortunato and Mengjie Liu and William H. Green and Klavs F. Jensen and Connor W. Coley},
  journal={Accounts of Chemical Research},
  year={2025},
  volume={58},
  number={11},
  pages={1764--1775},
  doi={10.1021/acs.accounts.5c00155},
  url={https://askcos.mit.edu}
}
```

### Retention Time Prediction

Retention times are predicted using the ReTiNA_XGB1 model from the ReTiNA retention time prediction series. The model accepts full LC-MS method parameters (gradient, column, flow rate, temperature) and molecular descriptors to predict retention times in seconds. The model file is in `rt_pred/ReTiNA_XGB1/`.

```
@modelcollection{retinamodels,
  title={ReTiNA-Models: Machine Learning Models for LC-MS Retention Time Prediction},
  author={Leung, Nathan},
  institution={Coley Research Group @ MIT},
  year={2025},
  howpublished={\url{https://huggingface.co/natelgrw/ReTiNA-Models}}
}
```

### Gradient Parameterization

Gradients are represented by 18 parameters that map to interpretable multi-step profiles:

**%B Parameters (0-100):**
- `b_0` to `b_9`: Ten independent %B values at successive time points
- No monotonicity constraints allow complex shapes (plateaus, reversals, multi-step)

**Time Spacing Parameters (0-1, auto-normalized):**
- `spacing_1` to `spacing_8`: Eight relative time intervals between points
- `t_0` fixed at 0 minutes, `t_9` fixed at method length
- Automatically normalized to distribute across total method duration

This creates flexible 10-point gradient profiles. The 18D parameterization allows the optimizer to discover non-obvious separation strategies while maintaining physical constraints.

### Separation Scoring

The objective function computes a weighted separation score based on:

1. **Pairwise Separations**: All compound pairs contribute based on their retention time differences, weighted by the product of their ASKCOS probabilities
2. **Spacing Reward**: Uses square-root scaling to reward larger separations while maintaining sensitivity
3. **Elution Penalties**: Compounds eluting outside the method window incur penalties

The score is normalized by the square root of the number of compound pairs to maintain comparable values across different reaction sizes.

### Optimization

Trust Region Bayesian Optimization (TuRBO) efficiently searches the 18-dimensional gradient space:

1. **Initial Sampling**: Random exploration with Latin hypercube sampling
2. **Gaussian Process Surrogate**: Models the separation score landscape with Matérn kernel
3. **Acquisition Function**: Upper Confidence Bound (UCB) balances exploration and exploitation
4. **Trust Regions**: Adaptive local search regions prevent over-exploration and focus on promising areas
5. **Restart Mechanism**: Escapes local optima when trust regions become too constrained

The optimizer typically converges within 100-200 evaluations depending on problem complexity.

### Output Format

The `optimize_gradient()` function generates an OptGradient JSON output:

```json
{
  "reactants": ["SMILES", ...],
  "solvent": "SMILES",
  "predicted_products": [
    {
      "smiles": "SMILES",
      "probability": 0.85,
      "mol_weight": 146.14
    }
  ],
  "optimized_gradient": [
    [0.00, 5.0],
    [1.25, 20.5],
    ...
    [12.00, 95.0]
  ],
  "gradient_params": [b_0, ..., b_9, spacing_1, ..., spacing_8],
  "separation_score": 0.152,
  "lcms_config": {
    "method_length": 12.0,
    "column": ["RP", 2.1, 100, 1.7],
    "flow_rate": 0.4,
    "temp": 45.0
  },
  "optimization_stats": {
    "n_evaluations": 100,
    "best_iteration": 84
  }
}
```


## 🪄 Citation

If you use Gradience in your research, please cite:

```
@software{gradience1.0.0,
  title={Gradience: Automated LC-MS Gradient Optimization},
  author={Leung, Nathan},
  institution={Coley Research Group @ MIT},
  year={2025},
  url={https://github.com/natelgrw/gradience}
}
```
