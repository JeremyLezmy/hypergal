
import os
import warnings
import requests


def download_urls(urls, fileout=None, client=None, **kwargs):
    """ downloads multiple urls into fileouts """
    if fileout is None:
        fileout = [None]*len(urls)
    elif len(fileout) != len(urls):
        raise ValueError(f"sizes of urls and fileout don't match ({len(urls)} vs. {len(fileout)})")
    
    if client is not None:
        from dask import delayed, distributed
        d_downloads = [delayed(download_single_url)(url_, fileout=fileout_, **kwargs)
                           for url_, fileout_ in zip(urls, fileout)]
        f_downloads = client.compute(d_downloads)
        return client.gather(f_downloads)

    # This could be multiprocessed.
    return [download_single_url(url_, fileout=fileout_, **kwargs)
                for url_, fileout_ in zip(urls, fileout)]
        
        
def download_single_url(url, fileout=None, mkdir=True,
                        overwrite=False, warn=True, chunk=1024, verbose=False,
                        **kwargs):
    """ Download the url target using requests.get.
    the data is returned (if fileout is None) or stored in `fileout`
    Pa
    """
    if (fileout is not None and fileout not in ["BytesIO", "StringIO"]) and not overwrite and os.path.isfile( fileout ):
        if warn:
            warnings.warn(f"{fileout} already exists: skipped")
        return None
    
    if verbose and fileout is not None:
        print("downloading {fileout}")

    if fileout is None:
        fileout = "BytesIO"
        
    request_fnc = "get" if not "data" in kwargs else "post"
    response = getattr(requests,request_fnc)(url, **kwargs)
    if response.status_code == 200:
        if fileout in ["BytesIO", "StringIO"]:
            import io
            return getattr(io, fileout)(response.content)
        
        with open(fileout, 'wb') as f:
            for data in response.iter_content(chunk):
                f.write(data)
    else:
        warnings.warn(f"Downloading problem: {response.status_code}")
        

