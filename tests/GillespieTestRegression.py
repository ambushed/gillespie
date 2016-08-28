import unittest
from gillespie import Gillespie
from gillespie import Setup

class GillespieTestSuiteRegression(unittest.TestCase):

    def testLotkaVolterra10paths15years(self):

        setup = Setup(yaml_file_name="models/lotka_volterra.yaml")
        propensities = setup.get_propensity_list()
        parameters = setup.get_parameter_list()
        species = setup.get_species()
        incr = setup.get_increments()
        nPaths = setup.get_number_of_paths()
        T = setup.get_time_horizon()
        my_gillespieUp = Gillespie(a=species[0], b=species[1], propensities=propensities, increments=incr, nPaths=nPaths,
                                   T=T, useSmoothing=False, numProc = 2)

        result = my_gillespieUp.run_simulation(*parameters)


        expectedMeanA = [1260.6, 1302.3, 961.9, 541.3, 299.6, 188.6, 136.8, 124.3, 133.9, 155.0, 187.9, 244.8, 343.8,
                         491.9, 705.8, 966.2, 1214.1, 1288.5, 1057.2, 621.9, 319.8, 174.2, 125.5, 116.0, 123.4, 141.5,
                         180.1,
                         247.7, 341.5, 495.4, 702.6, 973.0, 1198.4, 1244.9, 995.5, 628.2, 339.7, 198.0, 140.5, 118.4, 119.9,
                         142.5, 172.3, 235.0, 325.3, 455.4, 633.2, 844.5, 1030.5, 1125.0]

        expectedMeanB = [738.4, 1198.7, 1791.6, 2062.1, 1937.4, 1651.5, 1344.0, 1074.0, 852.0, 690.2, 568.9, 476.0,
                         413.0, 397.1, 423.4, 516.9, 751.4, 1183.3, 1702.3, 2065.2, 2018.9, 1731.3, 1397.0, 1094.1, 863.2,
                         694.5, 561.6, 472.2, 416.5, 392.8, 414.9, 519.3, 768.8, 1195.4, 1691.5, 1977.1, 1961.0, 1705.5,
                         1389.6, 1107.8, 875.6, 699.9, 571.1, 477.4, 419.6, 396.5, 428.4, 518.7, 719.4, 1016.8]

        aMean = result[:50]
        bMean = result[50:]
        self.assertListEqual(aMean, expectedMeanA)
        self.assertListEqual(bMean, expectedMeanB)

    def testLotkaVolterraAutoGrad(self):

        setup = Setup(yaml_file_name="models/lotka_volterra.yaml")
        propensities = setup.get_propensity_list()
        parameters = setup.get_parameter_list()
        species = setup.get_species()
        incr = setup.get_increments()
        nPaths = 2
        T = 2.0

        gillespieGrad = Gillespie(a=species[0],b=species[1],propensities=propensities,increments=incr, nPaths = nPaths,T=T,useSmoothing=True,numProc=2)
        gradient = gillespieGrad.take_gradients(*parameters)

        expected = [0.205909100363561, 0.20805536807560177, 0.20963661704449876, 0.21143004674883176,
                   0.21223841875244698, 0.21346284377271424, 0.21416575342524413, 0.21410798335165915,
                   0.21349333303106896, 0.21204227692038369, 0.21122829717797897, 0.20878900850345272,
                   0.20706928234608202, 0.20501739244067288, 0.20187689508842616, 0.19697261413659156,
                   0.19363807364611357, 0.18834792832928224, 0.18252652373565958, 0.17643699746163188,
                   0.16962195921877571, 0.16280595866068348, 0.15648563921143901, 0.15051363781790228,
                   0.14537464149599619, 0.14057828511832993, 0.13647236870514312, 0.13145925314805509,
                   0.12684497156143748, 0.12215968151207643, 0.11817093709820863, 0.11405767400197431,
                   0.11069650439278614, 0.10709993428037709, 0.10213484974070035, 0.1004140660330772,
                   0.098375735631237443, 0.093159998876577693, 0.087689484083106145, 0.08544590281358061,
                   0.082451799598521794, 0.080321426233540799, 0.075304245324723157, 0.075013978609624971,
                   0.074531804521655032, 0.075886338901131112, 0.072836943837557955, 0.071230026869074953,
                   0.07362651128777975, 0.072178198197482696, -0.04334562045423937, -0.045357554569389794,
                   -0.047199118331721676, -0.049302552588235349, -0.050545378732388783, -0.052025304435364461,
                   -0.053268307062119838, -0.053973153681620076, -0.054161155823272231, -0.053850288034399549,
                   -0.054345560118732525, -0.053491667208895227, -0.053252802181140271, -0.052957231396049867,
                   -0.051836620952625756, -0.04941673765383283, -0.047983197984209636, -0.045516286855536381,
                   -0.04238354902094546, -0.038891370504892969, -0.03475996155080234, -0.030863823601154416,
                   -0.026931605834836305, -0.0230154866914761, -0.019659395931897413, -0.016587740713791349,
                   -0.013686627029624237, -0.01034318143788166, -0.0071693180127194193, -0.0039982917633676653,
                   -0.001134086817833902, 0.0018262113772787476, 0.0044047201615518075, 0.0070128588898664343,
                   0.010259507439418766, 0.011968693899427977, 0.013339078329716321, 0.016437742013470855,
                   0.019391627235177213, 0.021018841932094186, 0.022745250164090756, 0.024163613949701762,
                   0.026032289510932026, 0.026889084950617429, 0.027769121172698013, 0.02854308624995339,
                   0.029664775555265435, 0.030319135982071206, 0.030946238576571125, 0.031767001147130255]

        expectedGradA = expected[:50]
        expectedGradB = expected[50:]

        aGradient = gradient[:50]
        bGradient = gradient[50:]

        self.assertListEqual(aGradient, expectedGradA)
        self.assertListEqual(bGradient, expectedGradB)

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(GillespieTestSuiteRegression())
    unittest.TextTestRunner().run(suite)


