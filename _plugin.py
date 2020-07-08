# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import wbia
from wbia.control import controller_inject, docker_control
from wbia.constants import ANNOTATION_TABLE
import wbia.constants as const
import utool as ut
import dtool as dt
import vtool as vt
import numpy as np
import base64
import requests
from PIL import Image
from io import BytesIO
import cv2
import tqdm


(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)
register_preproc_annot = controller_inject.register_preprocs['annot']


u"""
Interfacing with the ACR from python is a headache, so for now we will assume that
the docker image has already been downloaded. Command:

docker pull wildme.azurecr.io/wbia/kaggle7:latest
"""


BACKEND_URL = None


def _wbia_plugin_kaggle7_check_container(url):
    endpoints = {
        'api/classify': ['POST'],
    }
    flag_list = []
    endpoint_list = list(endpoints.keys())
    for endpoint in endpoint_list:
        print('Checking endpoint %r against url %r' % (endpoint, url,))
        flag = False
        required_methods = set(endpoints[endpoint])
        supported_methods = None
        url_ = 'http://%s/%s' % (url, endpoint,)

        try:
            response = requests.options(url_, timeout=1)
        except:
            response = None

        if response is not None and response.status_code:
            headers = response.headers
            allow = headers.get('Allow', '')
            supported_methods_ = [method.strip().upper() for method in allow.split(',')]
            supported_methods = set(supported_methods_)
            if len(required_methods - supported_methods) == 0:
                flag = True
        if not flag:
            args = (endpoint,)
            print(
                '[wbia_kaggle7 - FAILED CONTAINER ENSURE CHECK] Endpoint %r failed the check'
                % args
            )
            print('\tRequired Methods:  %r' % (required_methods,))
            print('\tSupported Methods: %r' % (supported_methods,))
        print('\tFlag: %r' % (flag,))
        flag_list.append(flag)
    supported = np.all(flag_list)
    return supported


docker_control.docker_register_config(
    None,
    'flukebook_kaggle7',
    'wildme.azurecr.io/wbia/kaggle7:latest',
    run_args={'_internal_port': 5000, '_external_suggested_port': 5000},
    container_check_func=_wbia_plugin_kaggle7_check_container,
)


@register_ibs_method
def wbia_plugin_kaggle7_ensure_backend(ibs, container_name='flukebook_kaggle7', **kwargs):
    global BACKEND_URL
    # make sure that the container is online using docker_control functions
    if BACKEND_URL is None:
        # Register depc blacklist
        prop_list = [None, 'theta', 'verts', 'species', 'name', 'yaws']
        for prop in prop_list:
            ibs.depc_annot.register_delete_table_exclusion('KaggleSevenChip', prop)
            ibs.depc_annot.register_delete_table_exclusion(
                'KaggleSevenIdentification', prop
            )

        BACKEND_URLS = ibs.docker_ensure(container_name)
        if len(BACKEND_URLS) == 0:
            raise RuntimeError('Could not ensure container')
        elif len(BACKEND_URLS) == 1:
            BACKEND_URL = BACKEND_URLS[0]
        else:
            BACKEND_URL = BACKEND_URLS[0]
            args = (
                BACKEND_URLS,
                BACKEND_URL,
            )
            print('[WARNING] Multiple BACKEND_URLS:\n\tFound: %r\n\tUsing: %r' % args)
    return BACKEND_URL


class KaggleSevenChipConfig(dt.Config):  # NOQA
    _param_info_list = [ut.ParamInfo('chip_padding', 32), ut.ParamInfo('ext', '.jpg')]


@register_preproc_annot(
    tablename='KaggleSevenChip',
    parents=[ANNOTATION_TABLE],
    colnames=['image', 'image_width', 'image_height'],
    coltypes=[dt.ExternType(vt.imread, vt.imwrite), int, int],
    configclass=KaggleSevenChipConfig,
    fname='kaggle7',
    chunksize=128,
)
def wbia_plugin_kaggle7_chip_depc(depc, aid_list, config):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_kaggle7._plugin --test-wbia_plugin_kaggle7_chip_depc
        python -m wbia_kaggle7._plugin --test-wbia_plugin_kaggle7_chip_depc:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_kaggle7._plugin import *  # NOQA
        >>> import wbia
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_kaggle7()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> images = ibs.depc_annot.get('KaggleSevenChip', aid_list, 'image')
        >>> image = images[0]
        >>> assert ut.hash_data(image) in ['imlkoiskkykpbwozghmpidlqwbmglzhw']
    """
    padding = config['chip_padding']

    tips_list = depc.get('Notch_Tips', aid_list)
    size_list = depc.get('chips', aid_list, ('width', 'height'))
    config_ = {
        'dim_size': 1550,
        'resize_dim': 'width',
        'ext': '.jpg',
    }
    chip_list = depc.get('chips', aid_list, 'img', config=config_, ensure=True)

    tps = cv2.createThinPlateSplineShapeTransformer()

    zipped = list(zip(aid_list, tips_list, size_list, chip_list))
    for aid, tip_list, size, chip in zipped:
        h0, w0, c0 = chip.shape
        notch = tip_list[0].copy()
        left = tip_list[1].copy()
        right = tip_list[2].copy()

        size = np.array(size, dtype=np.float32)
        notch /= size
        left /= size
        right /= size

        size = np.array([w0, h0], dtype=np.float32)
        notch *= size
        left *= size
        right *= size

        chip_ = chip.copy()
        h0, w0, c0 = chip_.shape

        left += padding
        notch += padding
        right += padding

        pad = np.zeros((h0, padding, 3), dtype=chip_.dtype)
        chip_ = np.hstack((pad, chip_, pad))
        h, w, c = chip_.shape
        pad = np.zeros((padding, w, 3), dtype=chip_.dtype)
        chip_ = np.vstack((pad, chip_, pad))
        h, w, c = chip_.shape

        delta = right - left
        radian = np.arctan2(delta[1], delta[0])
        degree = np.degrees(radian)
        M = cv2.getRotationMatrix2D((left[1], left[0]), degree, 1)
        chip_ = cv2.warpAffine(chip_, M, (w, h), flags=cv2.INTER_LANCZOS4)

        H = np.vstack((M, [0, 0, 1]))
        vert_list = np.array([notch, left, right])
        vert_list_ = vt.transform_points_with_homography(H, vert_list.T).T
        notch, left, right = vert_list_

        left[0] -= padding // 2
        left[1] -= padding // 2
        notch[1] += padding // 2
        right[0] += padding // 2
        right[1] -= padding // 2

        sshape = np.array([left, notch, right], np.float32)
        tshape = np.array([[0, 0], [w0 // 2, h0], [w0, 0]], np.float32)
        sshape = sshape.reshape(1, -1, 2)
        tshape = tshape.reshape(1, -1, 2)
        matches = [
            cv2.DMatch(0, 0, 0),
            cv2.DMatch(1, 1, 0),
            cv2.DMatch(2, 2, 0),
        ]
        tps.clear()
        tps.estimateTransformation(tshape, sshape, matches)
        chip_ = tps.warpImage(chip_)

        chip_ = chip_[:h0, :w0, :]
        chip_h, chip_w = chip_.shape[:2]

        yield (
            chip_,
            chip_w,
            chip_h,
        )


@register_route(
    '/api/plugin/kaggle7/chip/src/<aid>/',
    methods=['GET'],
    __route_prefix_check__=False,
    __route_authenticate__=False,
)
def kaggle7_chip_src(aid=None, ibs=None, **kwargs):
    from six.moves import cStringIO as StringIO
    from PIL import Image  # NOQA
    from flask import current_app, send_file
    from wbia.web import appfuncs as appf
    import six

    if ibs is None:
        ibs = current_app.ibs

    aid = int(aid)
    aid_list = [aid]
    chip_paths = ibs.depc_annot.get(
        'KaggleSevenChip', aid_list, 'image', read_extern=False, ensure=True
    )
    chip_path = chip_paths[0]

    # Load image
    assert chip_paths is not None, 'chip path should not be None'
    image = vt.imread(chip_path, orient='auto')
    image = appf.resize_via_web_parameters(image)
    image = image[:, :, ::-1]

    # Encode image
    image_pil = Image.fromarray(image)
    if six.PY2:
        img_io = StringIO()
    else:
        img_io = BytesIO()
    image_pil.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')


def get_b64_image_str(ibs, image_filepath, **kwargs):
    image = Image.open(image_filepath)
    byte_buffer = BytesIO()
    image.save(byte_buffer, format='JPEG')
    image_base64_str = base64.b64encode(byte_buffer.getvalue()).decode('utf-8')
    return image_base64_str


@register_ibs_method
def wbia_plugin_kaggle7_identify_aid(ibs, kchip_filepath, config={}, **kwargs):
    url = ibs.wbia_plugin_kaggle7_ensure_backend(**kwargs)
    image_base64_str = get_b64_image_str(ibs, kchip_filepath, **config)
    data = {'image': image_base64_str, 'config': config}
    url_ = 'http://%s/api/classify' % (url)
    # print('Sending identify to %s' % url)
    response = requests.post(url_, json=data, timeout=120)
    assert response.status_code == 200
    response = response.json()
    response = response.get('scores', None)
    return response


class KaggleSevenIdentificationConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('model_tag', 'crc'),
        ut.ParamInfo('k', 100),  # Return top-k results
    ]


@register_preproc_annot(
    tablename='KaggleSevenIdentification',
    parents=['KaggleSevenChip'],
    colnames=['response'],
    coltypes=[dict],
    configclass=KaggleSevenIdentificationConfig,
    fname='kaggle7',
    chunksize=128,
)
def wbia_plugin_kaggle7_identification_depc(depc, kchip_rowid_list, config):
    r"""
    Refine localizations for CurvRank with Dependency Cache (depc)

    CommandLine:
        python -m wbia_kaggle7._plugin --test-wbia_plugin_kaggle7_identification_depc
        python -m wbia_kaggle7._plugin --test-wbia_plugin_kaggle7_identification_depc:0

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia_kaggle7._plugin import *  # NOQA
        >>> import wbia
        >>> import numpy as np
        >>> from wbia.init import sysres
        >>> dbdir = sysres.ensure_testdb_kaggle7()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_image_aids(1)
        >>> response_list = ibs.depc_annot.get('KaggleSevenIdentification', aid_list, 'response')
        >>> response = response_list[0]
        >>> name_text_list = ibs.get_annot_names(aid_list)
        >>> name_text = name_text_list[0]
        >>> score = response.get(name_text, -1)
        >>> assert np.isclose(score, 1.0)
    """
    ibs = depc.controller

    kchip_filepath_list = depc.get_native(
        'KaggleSevenChip', kchip_rowid_list, 'image', read_extern=False
    )

    model_tag = config['model_tag']
    topk = config['k']
    config_ = {
        'model_tag': model_tag,
        'topk': topk,
    }
    for kchip_filepath in tqdm.tqdm(kchip_filepath_list):
        response = ibs.wbia_plugin_kaggle7_identify_aid(kchip_filepath, config=config_)
        yield (response,)


def get_match_results(depc, qaid_list, daid_list, score_list, config):
    """ converts table results into format for ipython notebook """
    # qaid_list, daid_list = request.get_parent_rowids()
    # score_list = request.score_list
    # config = request.config

    unique_qaids, groupxs = ut.group_indices(qaid_list)
    # grouped_qaids_list = ut.apply_grouping(qaid_list, groupxs)
    grouped_daids = ut.apply_grouping(daid_list, groupxs)
    grouped_scores = ut.apply_grouping(score_list, groupxs)

    ibs = depc.controller
    unique_qnids = ibs.get_annot_nids(unique_qaids)

    # scores
    _iter = zip(unique_qaids, unique_qnids, grouped_daids, grouped_scores)
    for qaid, qnid, daids, scores in _iter:
        dnids = ibs.get_annot_nids(daids)

        # Remove distance to self
        annot_scores = np.array(scores)
        daid_list_ = np.array(daids)
        dnid_list_ = np.array(dnids)

        is_valid = daid_list_ != qaid
        daid_list_ = daid_list_.compress(is_valid)
        dnid_list_ = dnid_list_.compress(is_valid)
        annot_scores = annot_scores.compress(is_valid)

        # Hacked in version of creating an annot match object
        match_result = wbia.AnnotMatch()
        match_result.qaid = qaid
        match_result.qnid = qnid
        match_result.daid_list = daid_list_
        match_result.dnid_list = dnid_list_
        match_result._update_daid_index()
        match_result._update_unique_nid_index()

        grouped_annot_scores = vt.apply_grouping(annot_scores, match_result.name_groupxs)
        name_scores = np.array([np.sum(dists) for dists in grouped_annot_scores])
        match_result.set_cannonical_name_score(annot_scores, name_scores)
        yield match_result


class KaggleSevenConfig(dt.Config):  # NOQA
    """
    CommandLine:
        python -m wbia_kaggle7._plugin --test-KaggleSevenConfig

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia_kaggle7._plugin import *  # NOQA
        >>> config = KaggleSevenConfig()
        >>> result = config.get_cfgstr()
        >>> print(result)
        KaggleSeven(dim_size=2000)
    """

    def get_param_info_list(self):
        return []


class KaggleSevenRequest(dt.base.VsOneSimilarityRequest):
    _symmetric = False
    _tablename = 'KaggleSeven'

    @ut.accepts_scalar_input
    def get_fmatch_overlayed_chip(request, aid_list, config=None):
        depc = request.depc
        ibs = depc.controller
        chip_paths = ibs.depc_annot.get(
            'KaggleSevenChip',
            aid_list,
            'image',
            config=config,
            read_extern=False,
            ensure=True,
        )
        chips = list(map(vt.imread, chip_paths))
        return chips

    def render_single_result(request, cm, aid, **kwargs):
        # HACK FOR WEB VIEWER
        chips = request.get_fmatch_overlayed_chip([cm.qaid, aid], config=request.config)
        import vtool as vt

        out_img = vt.stack_image_list(chips)
        return out_img

    def postprocess_execute(request, parent_rowids, result_list):
        qaid_list, daid_list = list(zip(*parent_rowids))
        score_list = ut.take_column(result_list, 0)
        depc = request.depc
        config = request.config
        cm_list = list(get_match_results(depc, qaid_list, daid_list, score_list, config))
        return cm_list

    def execute(request, *args, **kwargs):
        # kwargs['use_cache'] = False
        result_list = super(KaggleSevenRequest, request).execute(*args, **kwargs)
        qaids = kwargs.pop('qaids', None)
        if qaids is not None:
            result_list = [result for result in result_list if result.qaid in qaids]
        return result_list


@register_preproc_annot(
    tablename='KaggleSeven',
    parents=[ANNOTATION_TABLE, ANNOTATION_TABLE],
    colnames=['score'],
    coltypes=[float],
    configclass=KaggleSevenConfig,
    requestclass=KaggleSevenRequest,
    fname='kaggle7',
    rm_extern_on_delete=True,
    chunksize=None,
)
def wbia_plugin_kaggle7(depc, qaid_list, daid_list, config):
    r"""
    CommandLine:
        python -m wbia_kaggle7._plugin --exec-wbia_plugin_kaggle7
        python -m wbia_kaggle7._plugin --exec-wbia_plugin_kaggle7:0

    Example0:
        >>> # DISABLE_DOCTEST
        >>> from wbia_kaggle7._plugin import *
        >>> import wbia
        >>> import itertools as it
        >>> import utool as ut
        >>> from wbia.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_kaggle7()
        >>> ibs = wbia.opendb(dbdir=dbdir)
        >>> depc = ibs.depc_annot
        >>> gid_list = ibs.get_valid_gids()[:1]
        >>> aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        >>> annot_name_list = ibs.get_annot_names(aid_list)
        >>> aid_list_ = ibs.add_annots(gid_list, [(0, 0, 1, 1)] * len(gid_list), name_list=annot_name_list)
        >>> qaid_list = aid_list
        >>> daid_list = ibs.get_valid_aids()
        >>> root_rowids = tuple(zip(*it.product(qaid_list, daid_list)))
        >>> config = KaggleSevenConfig()
        >>> # Call function via request
        >>> request = KaggleSevenRequest.new(depc, qaid_list, daid_list)
        >>> result = request.execute()
        >>> ibs.delete_annots(aid_list_)
        >>> am = result[0]
        >>> unique_nids = am.unique_nids
        >>> name_score_list = am.name_score_list
        >>> unique_name_text_list = ibs.get_name_texts(unique_nids)
        >>> name_score_list_ = ['%0.04f' % (score, ) for score in am.name_score_list if score >= 0.0001]
        >>> name_score_dict = dict(zip(unique_name_text_list, name_score_list_))
        >>> print('Queried KaggleSeven algorithm for ground-truth ID = %s' % (annot_name_list, ))
        >>> result = ut.repr3(name_score_dict)
        >>> print(result)
        {
            '51649bb0-0031-4866-8ed5-f543883f9cb8': '1.0000',
        }
    """
    ibs = depc.controller

    qaids = list(set(qaid_list))
    daids = list(set(daid_list))

    assert len(qaids) == 1
    response_list = ibs.depc_annot.get('KaggleSevenIdentification', qaids, 'response')
    response = response_list[0]

    if response is None:
        response = {}

    names = ibs.get_annot_name_texts(daids)
    name_counter_dict = {}
    for daid, dname in zip(daids, names):
        if dname in [None, const.UNKNOWN]:
            continue
        if dname not in name_counter_dict:
            name_counter_dict[dname] = 0
        name_counter_dict[dname] += 1

    name_score_dict = {}
    name_list = response.keys()
    for name in name_list:
        name_score = response[name]
        name_score = round(name_score, 4)
        name_counter = name_counter_dict.get(name, 0)
        if name_counter <= 0:
            if name_score > 0.01:
                args = (
                    name,
                    name_score,
                    len(daids),
                )
                print(
                    'Suggested match name = %r with score = %0.04f is not in the daids (total %d)'
                    % args
                )
            continue
        assert name_counter >= 1
        annot_score = name_score / name_counter
        name_score_dict[name] = annot_score

    dname_list = ibs.get_annot_name_texts(daid_list)
    for qaid, daid, dname in zip(qaid_list, daid_list, dname_list):
        value = name_score_dict.get(dname, 0)
        yield (value,)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_kaggle7._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
