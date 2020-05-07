
def parse_line(line):
    """
    Parse this like:

    'Add:O Don:B-entity_name and:I-entity_name Sherri:I-entity_name to:O 
    my:B-playlist_owner Meditate:B-playlist to:I-playlist Sounds:I-playlist
     of:I-playlist Nature:I-playlist playlist:O <=> AddToPlaylist'

    to:
    
    {'intent_label': 'AddToPlaylist',
    'length': xx
    'word_labels': 'O B-entity_name I-entity_name I-entity_name O',
    'words': 'Add Don and Sherri to my Meditate to Sounds of Nature playlist'}
    """
    utterance_data, intent_label = line.split(" <=> ")
    items = utterance_data.split()
    words = [item.rsplit(":", 1)[0]for item in items]
    word_labels = [item.rsplit(":", 1)[1]for item in items]
    return {
        "intent_label": intent_label,
        "words": " ".join(words),
        "word_labels": " ".join(word_labels),
        "length": len(words),
    }