'''
Script to render meshes
Relies on obj2png and model-converter-python repos (CITE)
Write so that one can import this function: take in a .off file and output a .png
    intermediary converts to a .obj
'''


mesh_path = "/om/user/katiemc/occupancy_networks/out/unconditional/onet_multi_simple/generation/meshes/03001627/d6edce467efdc48eba18ade30e563d37.off"

# python model-converter-python/convert.py -i=out/unconditional/onet_multi_simple/generation/meshes/03001627/cbc76d55a04d5b2e1d9a8cea064f5297.off -o=./sample.obj

# side view: python3 obj2png/src/obj2png.py -i sample.obj -o sample.png -a 270 -e 90
# some bottom: python3 obj2png/src/obj2png.py -i sample.obj -o sample.png -a 270 -e 40
# some aspect of front + depth

# could use python3 obj2png/src/obj2png.py -i sample2.obj -o sample2.png -a 0 -e 0
#   and rotate -45 degrees


python model-converter-python/convert.py -i=out/unconditional/chairs1/generation/meshes/03001627/1007e20d5e811b308351982a6e40cf41.off -o=/om/user/katiemc/occupancy_networks/sample.obj