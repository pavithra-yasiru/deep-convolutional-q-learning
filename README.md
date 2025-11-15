# 🟨👾 Deep Convolutional Q-Learning for Pac-Man
A Reinforcement Learning project using **Deep Convolutional Q-Networks (DQN)** to train an AI agent to play **Pac-Man** using high-dimensional pixel observations.

---

## 📌 Project Overview
This project demonstrates how a Deep Q-Learning agent can be trained to play Pac-Man using **raw images** as input.  
To achieve this, we build:

- A **Convolutional Neural Network (CNN)** for feature extraction  
- A **Deep Q-Network (DQN)** to predict optimal actions  
- An **experience replay buffer**  
- An **epsilon-greedy exploration strategy**  
- A **Gymnasium-based Pac-Man environment**  

The notebook walks step-by-step through the full process: environment setup, model architecture, training loop, and evaluation.

---

## 📂 Files in This Repository
| File | Description |
|------|-------------|
| `Deep_Convolutional_Q_Learning_for_Pac_Man_Complete_Code.ipynb` | Complete notebook containing model architecture, training logic, and evaluation code |
| `README.md` | Project documentation (this file) |
| *(optional)* `results/video.mp4` | Sample gameplay recorded after training |

---

## 🛠️ Technologies Used
- **Python**
- **PyTorch**
- **Gymnasium**
- **NumPy**
- **OpenCV**
- **Matplotlib**

---

## 🧠 Model Architecture
The model is a **Convolutional Neural Network (CNN)** that processes sequential frames and outputs Q-values for each possible action.

**Key Components:**
- Three convolutional layers with ReLU activations  
- Fully connected layers mapping extracted features → Q-values  
- Optimized using **Adam optimizer** and **MSE Loss**  

---

## 🎮 Environment
The environment is built using **Gymnasium**, with:

- Preprocessing (resizing, grayscale, stacking frames)
- Action space mapping for Pac-Man
- Reward shaping for effective learning

---

## 🔄 Training Pipeline
The training process follows the classic DQN workflow:

1. Initialize replay memory and neural network  
2. For each episode:  
   - Reset environment  
   - For each step:
     - Select action using epsilon-greedy  
     - Observe next state & reward  
     - Store transition in replay buffer  
     - Sample batch from memory and update network  
3. Periodically update target network  
4. Save model checkpoints  

---

## 📈 Results
After sufficient training episodes, the agent:

- Learns to navigate the maze  
- Avoids ghosts more intelligently  
- Improves score over time  

Performance varies depending on training length and hyperparameters.

---

## ▶️ How to Run the Notebook

### **1. Install Dependencies**
```bash
pip install gymnasium
pip install "gymnasium[box2d]"
pip install torch torchvision torchaudio
pip install opencv-python
pip install matplotlib
