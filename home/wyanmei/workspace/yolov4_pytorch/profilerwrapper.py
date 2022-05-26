import timer
import memorizer
import datascale
import csv


class ProfilerWrapper(object):
    def __init__(self):
        self.mr = memorizer.MemRec()
        self.tt = timer.Clock()
        self.scale = datascale.DataScale()

    def report(self, sample=False):

        file = open("table_1.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["layer_name", "time", "output size", "mem consumption"])
        writer.writerow(["user_input", "nan", "%.4f" % (self.scale.scaleRec['user_input'] / (1024 ** 2)), "nan"])
        self.scale.scaleRec.pop('user_input')
        if self.mr.mem_cuda:
            for key, value in self.tt.time_records.items():
                writer.writerow([key,
                                 "%.4f" % min(self.tt.time_records[key]),
                                 "%.4f" % (self.scale.scaleRec[key] / (1024 ** 2)),
                                 "%.4f" % (sum(self.mr.mem_cuda[key]) / len(self.mr.mem_cuda[key]))])
        else:
            for key, value in self.tt.time_records.items():
                writer.writerow([key,
                                 "%.4f" % sum(self.tt.time_records[key]) / len(self.tt.time_records[key]),
                                 "%.4f" % (self.scale.scaleRec[key] / (1024 ** 2)),
                                 "%.4f" % (sum(self.mr.mem_cpu[key]) / len(self.mr.mem_cpu[key]))])
        file.close()
        file = open("table_2.csv", "w")
        writer = csv.writer(file)
        writer.writerow(["tensor name", "src", "dst"])
        for key, value in self.scale.dependencyRec.items():
            for dest in value.dest_list:
                writer.writerow([key,
                                 value.src,
                                 dest])
        file.close()

        # # =====================================
        # # Time
        # # =====================================
        # print("Average Time of Each Layer")
        # for key, value in self.tt.time_records.items():
        #     print("{:<15} {:<20}".format(key, sum(value) / len(value)))
        #
        # # =====================================
        # # Memory
        # # =====================================
        #
        # print("mem_self_cpu | Average Mem Consumption of Each Layer")
        # for key, value in self.mr.mem_self_cpu.items():
        #     print("{:<15} {:<20}".format(key, sum(value) / len(value)))
        #
        # print("mem_cpu | Average Mem Consumption of Each Layer")
        # for key, value in self.mr.mem_cpu.items():
        #     print("{:<15} {:<20}".format(key, sum(value) / len(value)))
        #
        # if self.mr.mem_cuda:
        #     print("mem_self_cuda | Average Mem Consumption of Each Layer")
        #     for key, value in self.mr.mem_self_cuda.items():
        #         print("{:<15} {:<20}".format(key, sum(value) / len(value)))
        #
        #     print("mem_cuda | Average Mem Consumption of Each Layer")
        #     for key, value in self.mr.mem_cuda.items():
        #         # value.pop(0)
        #         print("{:<15} {:<20}".format(key, sum(value) / len(value)))
        #
        # # =====================================
        # # Scale
        # # =====================================
        #
        # print("\nData size of each tensor: \n")
        # print("{:<15} {:<20}".format('Name', 'Size'))
        # print("================================================\n")
        # for key, value in self.scale.scaleRec.items():
        #     print("{:<15} {:<20}".format(key, value / (1024 ** 2)))
        #
        # print("\nData Dependency of each layer \n")
        # print("{:<20} {:<15} {:<15}".format('Name', 'Source', "Destination"))
        # print("================================================\n")
        # for key, value in self.scale.dependencyRec.items():
        #     for dest in value.dest_list:
        #         print("{:<20} {:<15} {:<15}".format(key, value.src, dest))
