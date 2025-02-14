import func as kd

initial_params = {
    "imfile": "imgs/axialBrain.jpg",
    "showProgress": False,''
    "loop": True,
    "sequenceType": "EPI",
    "noiseType": "none",
    "noiseScale": 0.5,
    "FOV": 180,
    "res": 2,
    "imSize": 180,
    "imRes": 1,
    "bandwidth": 125,
    "echoTime": 30,
    "oversample": 1
}

kd.kspace_demo(initial_params)