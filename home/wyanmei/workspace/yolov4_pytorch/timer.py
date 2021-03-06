import time
import matplotlib.pyplot as plt


class Clock(object):
    def __init__(self):
        super().__init__()
        self.time_records = {}
        self.time_starts = {}

    def tic(self, layername):
        self.time_starts[layername] = time.time()

    def toc(self, layername):
        diff = time.time() - self.time_starts[layername]
        if layername not in self.time_records:
            self.time_records[layername] = []
        self.time_records[layername].append(diff)

    def report(self, sample=False):
        plt.rcParams.update({'font.size': 8})
        if sample:
            for key, value in self.time_records.items():
                # print("{:<15} {:<20}".format(key, value))
                print(key, end=' :: ')
                print(value, flush=True)


        layernames = []
        avg_times = []
        print("Average Time of Each Layer")
        for key, value in self.time_records.items():
            # value.pop(0)
            print("{:<15} {:<20}".format(key, sum(value) / len(value)))
            # print(f"{key} :: {sum(value) / len(value)}")
            # layernames.append(key)
            # avg_times.append(sum(value) / len(value))

        # fig = plt.figure()
        # # creating the bar plot
        # plt.bar(layernames, avg_times, width=0.4)
        #
        # plt.xlabel("layernames")
        # plt.xticks(rotation=45, ha="right")
        # plt.ylabel("avg_times (s)")
        # plt.savefig("layer-avg_time.png")