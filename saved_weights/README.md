# Saved Model Weights

This folder contains the trained model checkpoints generated during experimentation and performance optimization of the OCTMNIST retinal disease classification task.

## Files

- **base_model.pth**  
  Weights of the baseline CNN model trained without additional optimization techniques.

- **improved_model.pth**  
  Weights of the optimized CNN model incorporating performance improvements such as regularization and learning rate tuning, etc. This is the final chosen optimized model with accuracy more than 80%.

- **improved_model_es.pth**  
  Weights of the optimized model trained with Early Stopping to prevent overfitting and improve generalization.

---

## Notes

- All models were trained using PyTorch.
- The best-performing model was selected based on validation performance.

