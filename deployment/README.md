## Project Structure

- **model.py**  
  Implements the custom Convolutional Neural Network (CNN) architecture built from scratch in PyTorch for multi-class OCT image classification.

- **improved_model.pth**  
  Serialized weights of the best-performing model (optimized using performance enhancement techniques such as regularization and learning rate tuning).

- **app.py**  
  Streamlit-based deployment script enabling real-time inference. Users can upload OCT images and receive predicted retinal disease classifications instantly.

- **requirements.txt**  
  Lists all required dependencies to reproduce training and deployment environments.

- **CHOROIDAL_NEOVASCULARIZATION.png / DRUSEN.png**  
  Sample OCT images used to validate model predictions and demonstrate deployment functionality.
