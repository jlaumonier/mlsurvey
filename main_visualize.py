import mlsurvey as mls


def main():
    app_interface = mls.sl.visualize.UserInterface('logs/')
    app_interface.run()


if __name__ == "__main__":
    main()
