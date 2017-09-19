class ResponseGen:
    def __init__(self):
        self.index_cls_name_mapper = dict()

    def map_index_to_cls_name(self, cls_index):
        return self.index_cls_name_mapper[cls_index]

    def gen_response(self, cls_index):
        return self.map_index_to_cls_name(cls_index=cls_index)
