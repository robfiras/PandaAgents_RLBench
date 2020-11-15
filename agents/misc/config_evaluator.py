import os
import sys
import getopt


def get_path_to_config(argv):
    """
    This functions reads the options and arguments
    :param argv:
    :return: either path to config_file or None
    """
    path_to_config = None

    try:
        opts, args = getopt.getopt(argv, "c:", ["config="])
    except getopt.GetoptError as e:
        print("\n Error: ", e, "\n")

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help_message()
            sys.exit()
        elif opt in ("-c", "--config"):
            path_to_config = arg

    # check if file exists
    if path_to_config and not os.path.isfile(path_to_config):
        print("File at ", path_to_config, " not found. Please double-check.")

    return path_to_config


def print_help_message():
    print("\nUsage: python main.py --config /path/to/your/config_file.yaml \n\n"
          "If no path is defined, the default configuration is used.")

