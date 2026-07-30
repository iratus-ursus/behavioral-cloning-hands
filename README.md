# Behavioral Cloning Hands


## Basic smoke tests

Runs a few episodes on each registered task and validates
observation / action specs, reward range and episode termination.

```bash
python test.py --headless
```

## Check Sim

### Headless (save video):

```bash
python check_sim.py --headless --enable_cameras --task reach_site --output results/reach_site.mp4
```

### Interactive (Isaac Sim GUI):

```bash
python check_sim.py --task reach_site --enable_cameras
# (omit --headless so the viewer window appears)
```

### Quick options:

```bash
python check_sim.py --headless --enable_cameras --num_steps 200 --fps 30
python check_sim.py --headless --enable_cameras --task lift_large_box --output results/lift.mp4
```

### What it checks

| Check                         | Description                                    |
| ----------------------------- | ---------------------------------------------- |
| Action bounds                 | Finite low / high                              |
| Observation keys & finiteness | Matches env.observation_spec                   |
| Reward                        | Finite values                                  |
| Multiple episodes             | Reset → step loop works                        |
| Camera (optional)             | `pixels` key exists and contains finite values |

## Train

### Classic reach (vision)

```bash
python train.py --algo ppo --task reach_site --headless --enable_cameras --num_envs 1024
```

### State-only

```bash
python train.py --algo ppo --task reach_site --headless --no_camera
```

### Other tasks (once you port the configs)

```bash
python train.py --algo ppo --task lift_large_box --headless --enable_cameras
```

### Behavioral Cloning

```bash
python train.py --algo bc --task reach_site --demo_path demos/expert.pt --headless --enable_cameras
```