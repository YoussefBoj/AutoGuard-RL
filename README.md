# 🚗 AutoGuard-RL

**Vision-Language Guided Safe Reinforcement Learning for Autonomous Driving**

A research-oriented autonomous driving system that combines Vision-Language Models (CLIP/BLIP), World Models (Dreamer-style RSSM), and Safe Reinforcement Learning to create safety-aware self-driving agents.

---

## 🌟 Features

- **🔒 Safety-First Learning**: Uses CLIP to assess scene safety and guide policy updates
- **🌍 World Model Imagination**: Predicts future states without environment interaction
- **🎯 Safe RL Agent**: Actor-Critic with VL-SAFE algorithm for balanced reward-safety optimization
- **🎮 Easy Testing**: Includes dummy environment for quick prototyping
- **📊 Real-time Monitoring**: TensorBoard integration for training visualization
- **🔧 Production Ready**: Modular, well-documented, and tested components

---

## 🏗️ Architecture

```
Camera Input (CARLA/BDD100K)
        ↓
  CLIP Encoder → Safety Risk Score (0-1)
        ↓
  World Model (RSSM) → Latent Dynamics + Imagination
        ↓
  Safe RL Agent → Action [steering, throttle]
        ↓
  Environment → Reward + Cost
        ↓
  Dashboard → Visualization
```

**Key Innovation**: Safety-aware policy weighting using vision-language models:
```
w = p(safe) × exp(β₁ × A_reward) + (1-p(safe)) × exp(-β₂ × A_cost)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/AutoGuard-RL.git
cd AutoGuard-RL

# Run setup script
bash setup.sh

# Install dependencies
pip install -r requirements.txt
```

### Test Components

```bash
python test_components.py
```

Expected output:
```
✓ CLIP Encoder working
✓ World Model working
✓ Safe Actor-Critic working
✓ Environment working
✓ Integration working
```

### Train Agent

```bash
# Quick test (10 episodes)
python train.py --episodes 10

# Full training (1000 episodes)
python train.py --episodes 1000

# Monitor with TensorBoard
tensorboard --logdir runs/
```

---

## 📁 Project Structure

```
AutoGuard-RL/
├── models/
│   ├── clip_encoder.py          # CLIP-based safety scorer
│   ├── rssm_worldmodel.py       # Dreamer-style world model
│   └── actor_critic_safe.py     # Safe RL agent (VL-SAFE)
├── utils/
│   ├── carla_env_wrapper.py     # Environment wrapper + replay buffer
│   ├── reward_functions.py      # Reward shaping logic
│   └── safety_monitor.py        # Safety violation tracking
├── config/
│   ├── model_config.yaml        # Model hyperparameters
│   └── train_config.yaml        # Training configuration
├── train.py                     # Main training script
├── test_components.py           # Component testing
└── requirements.txt             # Dependencies
```

---

## 🔧 Configuration

Edit `config/train_config.yaml` to customize training:

```yaml
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.0003
  safety_lambda: 10.0    # Safety penalty weight

environment:
  use_carla: false        # Set true for CARLA simulator
  image_size: [84, 84]
  max_episode_steps: 333
```

---

## 📊 Results

After 100 episodes with DummyEnv:

| Metric | Initial | After Training |
|--------|---------|----------------|
| Episode Reward | ~15 | ~50 |
| Safety Cost | ~0.5 | ~0.1 |
| Episode Length | ~45 | ~80 |

---

## 🧠 Core Components

### 1. CLIP Safety Encoder

Computes semantic similarity between driving scenes and unsafe text prompts:

```python
from models.clip_encoder import ClipEncoder

encoder = ClipEncoder()
safety_score = encoder.safety_score(image)  # Returns 0.0 (safe) to 1.0 (unsafe)
```

### 2. RSSM World Model

Learns latent dynamics for imagination-based planning:

```python
from models.rssm_worldmodel import WorldModel

model = WorldModel(config)
output = model(images, actions)
# Returns: reconstructed images, predicted rewards/costs
```

### 3. Safe Actor-Critic Agent

Optimizes policy with safety-aware weighting:

```python
from models.actor_critic_safe import SafeActorCritic

agent = SafeActorCritic(config)
action = agent.select_action(state)  # Returns [steering, throttle]
```

---

## 📚 Research Foundation

This project builds on recent advances in autonomous driving research:

1. **VL-SAFE** - Vision-Language Guided Safety-Aware Reinforcement Learning (arXiv 2025)
2. **DreamerV3** - Mastering Diverse Domains through World Models (Hafner et al., 2023)
3. **SafeDreamer** - Safe Reinforcement Learning with World Models (ICLR 2024)
4. **DriveDreamer4D** - World Models for Driving Scene Representation (CVPR 2025)

---

## 🎯 Roadmap

- [x] Core components implementation
- [x] Dummy environment for testing
- [x] Training pipeline with TensorBoard
- [ ] Full CARLA integration
- [ ] BDD100K dataset preprocessing
- [ ] Multi-task learning
- [ ] Vision transformer backbone
- [ ] Streamlit dashboard

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

## 🙏 Acknowledgments

- [CARLA Simulator](https://carla.org/) for autonomous driving simulation
- [OpenAI CLIP](https://github.com/openai/CLIP) for vision-language understanding
- [DreamerV3](https://github.com/danijar/dreamerv3) for world model architecture
- [BDD100K](https://bdd-data.berkeley.edu/) for driving dataset



---

**Built with ❤️ for safer autonomous driving**
