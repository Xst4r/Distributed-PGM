from subprocess import *

#java -Xmx1g -classpath bin:lib/core/commons-math3-3.2.jar:lib/core/jayes.jar:lib/core/jgrapht-jdk1.6.jar:lib/extra/jgraphx.jar:lib/loader/weka.jar demo.Run
# dataFile pvalue imageOutputFile useGUI

def jarWrapper(*args):
    process = Popen(['java', '-jar']+list(args), stdout=PIPE, stderr=PIPE)
    ret = []
    while process.poll() is None:
        line = process.stdout.readline()
        if line != '' and line.endswith('\n'):
            ret.append(line[:-1])
    stdout, stderr = process.communicate()
    ret += stdout.split('\n')
    if stderr != '':
        ret += stderr.split('\n')
    ret.remove('')
    return ret
