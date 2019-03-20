import sys

import mlsurvey as mls


def main():
    vw = mls.VisualizationWorkflow(directory=sys.argv[1])
    vw.run()


if __name__ == "__main__":
    main()
