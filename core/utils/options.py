import argparse


def get_parser(desc, default_task=""):
    pass


def get_env_parser(parser):
    parser.add_argument("-n", "--num_agents", default=1, type=int,
                        help="Number of agent in the env.")
    return parser


def get_preprocessing_parser(default_task):
    parser = get_parser("Preprocessing", default_task)
    return parser


def get_valid_parser(parser):
    #parser = get_parser("Validation", default_task)
    parser.add_argument("--valid_steps_per_episode",
                        default=1000, type=int,
                        help="Number of steps to take per evaluation episode.")
    parser.add_argument("--valid_interval", default=10,
                        type=int,
                        help="Evaluate every time epoch % eval_interval = 0.")
    parser.add_argument("--num_valid_episodes", default=5,
                        type=int,
                        help="Evaluate over eval_episodes evaluation episodes.")
    return parser


def get_test_parser(default_task):
    parser = get_parser("Test", default_task)
    return parser


def get_marl_parser(parser):
    parser.add_argument("--self_play", required=False, action="store_true",
                        help="to use a single master PPO agent.")
    parser.add_argument("--separate_agents", required=False, action="store_true",
                        help="to use a N PPO agents.")
    return parser


def get_logging_parser(parser):
    parser.add_argument("--save_interval", default=10, type=int,
                        help="Save policies every time epoch % eval_interval = 0.")
    parser.add_argument("--log_interval", default=1, type=int,
                        help="Log results every time epoch % eval_interval = 0.")
    parser.add_argument("-tb", "--use_tensorboard", required=False,
                        action="store_true", help="Log with tensorboard as well.")
    parser.add_argument('--td_train_dir', type=str, help='',
                        default='./tb_train')
    parser.add_argument('--td_valid_dir', type=str, help='',
                        default='./tb_val')
    return parser


def get_checkpoint_parser(parser):
    parser.add_argument('--checkpoints_dir', type=str, help='',
                        default='./checkpoints')
    parser.add_argument('--max_checkpoints', type=int, help='',
                        default=5)
    return parser

def get_imitation_learning_parser(parser):
    parser.add_argument('--goal_weight', type=float, help='',
                        default=0.7)
    parser.add_argument('--imitation_weight', type=float, help='',
                        default=0.3)
    return parser


def get_optimization_parser(parser):
    parser.add_argument("-lr", "--learning_rate", default=5e-8, type=float,
                        help="Learning rate for PPO agent(s).")
    parser.add_argument("--init_lr", default=5e-3, type=float,
                                        help="Learning rate for PPO agent(s).")
    parser.add_argument('--decay', type=str, default="exponential",
                                        help='')
    parser.add_argument('--decay_steps', type=int, default=1000,
                                        help='')
    parser.add_argument('--decay_rate', type=float, default=5e-8,
                                        help='')
    return parser


def get_distributed_training_parser(parser):
    pass


def get_dataset_parser(parser):
    parser.add_argument('--data_path', type=str, default="/home/simulate-part/data/Water-3D",
                                        help='')
    parser.add_argument('--seq_length', type=int, default=10,
                                        help='')
    return parser


def get_train_parser():
    parser = argparse.ArgumentParser()
    parser = get_env_parser(parser)
    parser = get_marl_parser(parser)
    parser = get_logging_parser(parser)
    parser = get_optimization_parser(parser)
    parser = get_dataset_parser(parser)
    parser = get_valid_parser(parser)
    parser = get_checkpoint_parser(parser)

    # Learning
    parser.add_argument("--num_train_episodes", default=1000, type=int,
                        help="Number of epochs to train agent over.")
    parser.add_argument("--collect_steps_per_episode",
                        default=1000, type=int,
                        help="Number of steps to take per collection episode.")
    parser.add_argument("--eps", type=float, default=0.0,
                        help="Probability of training on the greedy policy for a"
                             "given episode")
    parser.add_argument("--num_warmup_episodes",
                        default=1000, type=int,
                        help="")
    parser.add_argument('--train_batch', type=int, default=1,
                        help='')
    parser.add_argument('--valid_batch', type=int, default=1,
                        help='')
    parser.add_argument('--test_batch', type=int, default=1,
                        help='')
    parser.add_argument('--replay_buffer_max_length', type=int, help='',
                        default=10_000)
    parser.add_argument('--replay_buffer_capacity', type=int, help='',
                        default=10_000)
    parser.add_argument('--min_reward', type=float, help='',
                        default=-1.)
    parser.add_argument('--max_reward', type=float, help='',
                        default=1.)
    parser.add_argument("--normalize", action="store_true",
                        help="")
    parser.add_argument("--actor_net_layer", default=(256, 256), type=tuple,
                        help="")
    parser.add_argument("--value_net_layer", default=(256, 256), type=tuple,
                        help="")
    parser.add_argument("--dropout_layer_params", default=(0.1, 0.1), type=tuple,
                        help="")
    return parser


def parse_args(parser, modify_parser):
    return parser.parse_args()
