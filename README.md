# RISE_Architecture
RISE: Remote Instrumentation Science Environment for Learning-based Image Analytics

The Remote Instrumentation Science Environment (RISE) is a cloud-edge framework designed to automate and optimize scientific experimentation involving remote instruments and image analytics. Traditional approaches to operating instruments such as scanning electron microscopes often rely on manual control and post-hoc image analysis, which can be inefficient, error-prone, and inconsistent. RISE eliminates these challenges by integrating advanced AI agents, secure cloudlet-based computing, and real-time human–machine collaboration.
Key Features

    Automated Instrument Control: Use reinforcement learning (RL) and imitation learning (IL) agents to dynamically tune parameters (e.g., zoom, focus, contrast) for optimal image acquisition.

    Cloudlet-based Edge Computing: Lightweight cloud-edge nodes securely bridge remote scientific instruments with data centers, enabling low-latency, high-throughput data processing.

    Human-in-the-Loop Chatbot Assistant: Provides real-time interactive feedback and guidance during experiment execution.

    Validated on CNT Imaging: RISE was tested on carbon nanotube (CNT) image experiments, demonstrating its ability to enhance image quality and accelerate material discovery.

Repository Structure

This repository provides datasets and agent implementations used in our experiments:

    Dataset/
    Contains the training and testing datasets for reinforcement learning (RL) and imitation learning (IL) agents.

    IL_Agent/
    Implementation of the Imitation Learning (IL) agent, which learns from expert demonstrations to optimize instrument parameters.

    RL_Agent/
    Implementation of multi-agent reinforcement learning agents, including:

        MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

        MAPPO (Multi-Agent Proximal Policy Optimization)
