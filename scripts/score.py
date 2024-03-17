def score(psnr, runtime, psnr_interp=22.695024):
    '''
    Scoring formula for Real-time SR.
    - psnr_interp: PSNR for Bicubic interpolation in the val/test set.

    The funtion nullifies methods that perform worst than Bicubic.
    If the runtime is <16ms, growth is faster.
    Methods like RFDN (135.99ms) or IMDN (170.10ms) are not competitive.

    More info at: https://github.com/eduardzamfir/NTIRE23-RTSR#performance-of-baseline-methods
    Metric inspired by: https://arxiv.org/pdf/2211.05910.pdf
    '''

    diff = max(psnr - psnr_interp, 0)
    if diff == 0:
        return 0
    else:
        cte = 0.1
        return ((2 ** diff * 2)) / (cte * ((runtime ** 0.5)))


if __name__ == '__main__':
    print(score(psnr=23.102769, runtime=3.98))
