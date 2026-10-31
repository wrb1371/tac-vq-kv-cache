"""One complete-video raw decode cache; a common execution optimization only.

Question text and all processor/tokenizer work remain per question. Returned
audio/video objects are deep copied so processor mutation cannot alter a
later question. No video is truncated, transcoded, resampled or substituted.
"""
import copy,json,pathlib

def install_media_cache():
    import inference.inputs as native
    if hasattr(native,'_r13_original_process_mm_info'):return native._r13_media_cache_stats
    original=native.process_mm_info
    state={'key':None,'payload':None};stats={'misses':0,'hits':0,'capacity_complete_videos':1}
    def cached(conversation,**kwargs):
        media=[entry for turn in conversation for entry in turn.get('content',[]) if entry.get('type') in ['video','image','audio']]
        assert len(media)==1 and media[0]['type']=='video','this bounded cache supports the native single-video interface'
        source=pathlib.Path(media[0]['video']);stat=source.stat()
        key=json.dumps({'media':media,'kwargs':kwargs,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns},sort_keys=True)
        if key==state['key']:
            stats['hits']+=1;return copy.deepcopy(state['payload'])
        payload=original(conversation,**kwargs);state.update(key=key,payload=copy.deepcopy(payload));stats['misses']+=1
        return payload
    native._r13_original_process_mm_info=original
    native.process_mm_info=cached;native._r13_media_cache_stats=stats
    return stats
