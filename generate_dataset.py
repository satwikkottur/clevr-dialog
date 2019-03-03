'''Script to generate CLEVR-Dialog dataset.

Needs access to the following files:
synonyms:
caption templates:
question templates:

Usage:

Author: Satwik Kottur
'''


import copy
import collections
import json
import multiprocessing
import os
import pdb
import random
import re
import time

from absl import flags
from absl import app
import numpy as np
from tqdm import tqdm as progressbar

from util import clevr as utils
from util import constraints as constraints


flags.DEFINE_string('synonym_path', 'synonyms.json', 'Path to synonyms file')
flags.DEFINE_string('metainfo_path', 'metainfo.json',
                    'Path to meta information file')
flags.DEFINE_string('caption_template_root', 'templates/captions/',
                    'Root to folder with caption templates')
flags.DEFINE_string('question_template_root', 'templates/questions/',
                    'Root to folder with question templates')
flags.DEFINE_string('scene_path', 'scenes/CLEVR_train_scenes.json',
                    'Path to CLEVR scene path json file')
flags.DEFINE_string('scene_id_file', '',
                    'Path to specific CLEVR scene ids to generate dialogs')
flags.DEFINE_string('save_path', 'dialogs_gen_debug.json',
                    'Path to save the dataset json')
flags.DEFINE_integer('num_beams', 100, 'Number of beams in dialog search')
flags.DEFINE_integer('num_workers', 1, 'Number of workers to use in search')
flags.DEFINE_integer('captions_per_image', 5, 'Number of captions per image')
flags.DEFINE_integer('num_images', -1,
                     'Number of images to generate dialogs. -1 for all.')
flags.DEFINE_integer('num_rounds', 10, 'Number of rounds in each dialog')

FLAGS = flags.FLAGS


# relations
relations = {"left": ["left of", "to the left of", "on the left side of"],
             "right": ["right of", "to the right of", "on the right side of"],
             "behind": ["behind"],
             "front": ["in front of"]}

# independent question templates
indep_ques_templates = ['count-all', 'count-other', 'count-attribute',
                        'exist-other', 'exist-attribute']

tag_map = {'<S>': 'shape', '<M>': 'material', '<Z>': 'size', '<C>': 'color',
           '<X>': 'count', '<R>': 'relation', '<A>': 'attribute'}
tag_inv_map = {attribute: tag for tag, attribute in tag_map.items()}
tag_map['<P>'] = 'relation' # add after the inverse map

attributes = ['size', 'color', 'material', 'shape']


def mapping(tag):
  """Map tag to attribute.
  """
  return tag_map[tag.replace('1', '')]


def inv_mapping(attribute, arg_id=0):
  """Inverse map attribute to tag.

  Args:
    attribute: Name of the attribute
    arg_id: Argument id to use. Append 1 if arg_id is 1, else nothing

  Returns:
    base_tag: The string for the tag
  """

  base_tag = tag_inv_map[attribute]
  if arg_id > 0:
    base_tag = base_tag[:-1] + str(arg_id) + base_tag[-1]

  return base_tag


# get the group id from tag
def get_tag_group(tag):
  """Get the group id from tag string.

  For example, tag string of <S> is 0, <S1> is 1.
  Assumes single digit group id.

  Args:
    tag: Tag string

  Returns:
    group_id: Return extracted group id
  """

  group_id = 0 if len(tag) <= 3 else int(tag[-2])
  return group_id

# global configs -- sampling 1/2/3 attributes respectively
probs = [0.70, 0.30, 0.00]

# number of beams, and distribution of question types
# start cutting down threads after 5th round
# heuristics (for round 4):
# A. count <= 2  1 <= seek <= 3  exist <= 2
# B. count + exist <= 3
# C. Independent questions <= 1
# heuristics (for round 5):
# A. count <= 2  2 <= seek <= 4  exist <= 2
# B. count + exist <= 3
# C. Independent questions <= 1
ranges = {3: {'indep': [0, 1], 'seek': [1, 4], 'exist': [0, 1],
              'count': [0, 1], 'exist+count': [0, 2]},
          4: {'indep': [0, 1], 'seek': [2, 4], 'exist': [0, 1],
              'count': [0, 1], 'exist+count': [0, 2]},
          5: {'indep': [0, 1], 'seek': [2, 5], 'exist': [0, 2],
              'count': [0, 2], 'exist+count': [0, 3]},
          6: {'indep': [0, 1], 'seek': [2, 5], 'exist': [0, 2],
              'count': [0, 2], 'exist+count': [0, 3]},
          7: {'indep': [0, 2], 'seek': [3, 5], 'exist': [0, 2],
              'count': [0, 2], 'exist+count': [0, 3]},
          8: {'indep': [0, 2], 'seek': [3, 6], 'exist': [0, 3],
              'count': [0, 3], 'exist+count': [0, 3]},
          9: {'indep': [0, 2], 'seek': [3, 6], 'exist': [0, 3],
              'count': [0, 3], 'exist+count': [0, 4]}}


# If shape is to be replaced, use 'thing' instead.
def replace_attribute(text, tag, obj_group, eliminate=False):
  group = get_tag_group(tag)

  if mapping(tag) == 'relation':
    # actual relation tag, else position tag
    if tag == '<R>':
      relation_cand = random.choice(relations[obj_group['relation']])
    else:
      relation_cand = obj_group['relation']

    return text.replace(tag, relation_cand)

  if mapping(tag) == 'shape':
    if eliminate:
      replacer = 'thing'
    else:
      replacer = str(obj_group['objects'][group][mapping(tag)])

    # plural forms for groups
    #if len(obj_group['objects']) > 1:
    if obj_group.get('count', 1) > 1 or obj_group.get('use_plural', False):
      replacer += 's'
  elif mapping(tag) == 'count':
    if eliminate:
      replacer = ''
    else:
      replacer = str(obj_group['count'])
  else:
    if eliminate:
      replacer = ''
    else:
      replacer = str(obj_group['objects'][group][mapping(tag)])

  return text.replace(tag, replacer)


def realize_text_and_extract_scene(scene, template, filter_objs):
  default_list = lambda: collections.defaultdict(list)
  graph = {'relationships': collections.defaultdict(default_list),
           'counts': {}, 'exists': {}, 'history': [], 'objects': {}}

  # number of inputs
  n_inputs = template.get('inputs', 1)
  # sample a text template
  text_sample = random.choice(template['text'])
  text_sample_index = template['text'].index(text_sample)

  # extract attribute tags and get them into groups
  tags = re.findall('(<[\d\w]*>)', text_sample)

  tag_groups = collections.defaultdict(list)
  for tag in tags:
    group_id = get_tag_group(tag)
    tag_groups[group_id].append(tag)

  # sample a random element from filtered
  arg_sample = random.choice(filter_objs)
  # scene information obtained from the current round
  graph_item = arg_sample['graph']

  # remove tags from text not allowed by filter_objs
  for arg_ind in range(n_inputs):
    obj_sample = arg_sample['objects'][arg_ind]
    avail_attrs = obj_sample['optional'] + obj_sample['required']

    for ii in tag_groups[arg_ind][::-1]:
      if mapping(ii) not in avail_attrs:
        tag_groups[arg_ind].remove(ii)
        text_sample = replace_attribute(text_sample, ii, arg_sample, True)

    # assert that all required attributes are present as tags
    for attribute in obj_sample['required']:
      required_tag = inv_mapping(attribute, arg_ind)
      try:
        assert required_tag in tag_groups[arg_ind], \
            'A required attribute is missing in template!'
      except: pdb.set_trace()

    # start compiling tags to keep
    tags_to_keep = [inv_mapping(ii, arg_ind) for ii in obj_sample['required']]

    # filter out those not present in text template
    optional_tags = [inv_mapping(ii,arg_ind) for ii in obj_sample['optional']]
    optional_tags = [ii for ii in optional_tags if ii in tag_groups[arg_ind]]

    # if tags_to_keep is empty, sample from optional with 1:70 2:25  3:5
    if len(optional_tags) > 0:
      if len(tags_to_keep) > 0:
        n_tags_sample = [0, 1, 2]
      else: n_tags_sample = [1, 2, 3]
      n_sample = np.random.choice(n_tags_sample, 1, p=probs, replace=False)
      # lower cap at the length of optional
      n_sample = min(n_sample[0], len(optional_tags))
      if n_sample > 0:
        tags_to_keep += random.sample(optional_tags, n_sample)

    # now create a dictionary of placeholders with actual attribute values
    for tag in tag_groups[arg_ind]:
      remove = tag not in tags_to_keep
      text_sample = replace_attribute(text_sample, tag, arg_sample, remove)

    # remove attributes from objects not included in tags_to_keep
    if 'objects' in graph_item:
      for ii in attributes:
        if inv_mapping(ii, arg_ind) not in tags_to_keep:
          if ii in graph_item['objects'][arg_ind]:
            del graph_item['objects'][arg_ind][ii]

  # record the caption info
  graph_item['round'] = 0

  sample = {}
  sample['template_info'] = [copy.deepcopy(template)]
  del sample['template_info'][-1]['text']
  sample['template_info'][-1]['index'] = text_sample_index
  sample['caption'] = text_sample
  sample['dialog'] = []

  # append history, update scene graph, and save the new scene graph
  graph['history'].append(graph_item)
  sample['graph'] = utils.merge_update_scene_graph(graph, graph_item)

  return sample


def realize_question(dialog, template, filter_objs):
  ## number of inputs
  n_inputs = template.get('inputs', 0)
  # sample a text template
  text_sample = random.choice(template['text'])
  text_sample_index = template['text'].index(text_sample)

  # extract attribute tags and get them into groups
  tags = re.findall('(<[\d\w]*>)', text_sample)

  tag_groups = collections.defaultdict(list)
  for tag in tags:
    group_id = get_tag_group(tag)
    tag_groups[group_id].append(tag)

  ## sample a random element from filtered
  arg_sample = random.choice(filter_objs)

  ## remove tags from text not allowed by filter_objs
  for arg_ind in range(n_inputs):
    obj_sample = arg_sample['objects'][arg_ind]
    avail_attrs = obj_sample['optional'] + obj_sample['required']

    for ii in tag_groups[arg_ind][::-1]:
      if mapping(ii) not in avail_attrs:
        tag_groups[arg_ind].remove(ii)
        text_sample = replace_attribute(text_sample, ii, arg_sample, True)

    # assert that all required attributes are present as tags
    for attribute in obj_sample['required']:
      required_tag = inv_mapping(attribute, arg_ind)
      # make an exception for <R> and <P>
      if required_tag == '<R>' and '<P>' in tag_groups[arg_ind]:
        continue

      try:
        assert required_tag in tag_groups[arg_ind], \
          'A required attribute is missing in template!'
      except: pdb.set_trace()

    # start compiling tags to keep
    tags_to_keep = [inv_mapping(ii, arg_ind) for ii in obj_sample['required']]

    # filter out those not present in text template
    optional_tags = [inv_mapping(ii,arg_ind) for ii in obj_sample['optional']]
    optional_tags = [ii for ii in optional_tags if ii in tag_groups[arg_ind]]

    # if tags_to_keep is empty, sample from optional with 1:70 2:25  3:5
    if len(optional_tags) > 0:
      if len(tags_to_keep) > 0:
        n_tags_sample = [0, 1, 2]
      else: n_tags_sample = [1, 2, 3]
      n_sample = np.random.choice(n_tags_sample, 1, p=probs, replace=False)
      # lower cap at the length of optional
      n_sample = min(n_sample[0], len(optional_tags))
      if n_sample > 0:
        tags_to_keep += random.sample(optional_tags, n_sample)

    # now create a dictionary of placeholders with actual attribute values
    for tag in tag_groups[arg_ind]:
      remove = tag not in tags_to_keep
      text_sample = replace_attribute(text_sample, tag, arg_sample, remove)

  # record info and merge scene graphs
  dialog_datum = {'question': text_sample, 'answer': arg_sample['answer'],
                  'template': template['label'], }
  dialog['template_info'].append(template.copy())
  del dialog['template_info'][-1]['text']
  dialog['template_info'][-1]['index'] = text_sample_index

  dialog['dialog'].append(dialog_datum)
  graph_item = arg_sample['graph']

  # if mergeable, add it to the objects list
  dialog['graph'] = utils.merge_update_scene_graph(dialog['graph'], graph_item)

  # if there are volatile objects in the graph item, remove them
  for obj in graph_item['objects'][::-1]:
    if obj.get('volatile', False): graph_item['objects'].remove(obj)
  dialog['graph']['history'].append(graph_item)

  return dialog


# method to clean the text
def clean_text_subroutine(text, thing, suffix):
  # synonyms + skipping optional part of the sentence
  text = skip_and_replace_phrases(text)

  # Remove full stop, empty spaces, capitalize the start letter.
  text = re.sub(' +', ' ', text.replace(suffix, '').strip(' '))
  # First replace 'a thing' -> 'an object'.
  # Then perform remaining actions.
  if thing == 'object':
    text = text.replace('a thing', 'an object')
  text = text.replace('thing', thing)
  text = text[0].upper() + text[1:] + suffix

  return text


# method to clean the text
def clean_dialog_text(dialogs):
  # replace thing with object throughout with probability 0.5
  thing = 'thing' if random.random() > 0.5 else 'object'

  for index, dialog_datum in enumerate(dialogs):
    # clean the caption
    text = dialog_datum['caption']
    dialogs[index]['caption'] = clean_text_subroutine(text, thing, '.')

    for r_id, dialog in enumerate(dialog_datum['dialog']):
      # clean the caption
      text = dialog['question']
      text = clean_text_subroutine(text, thing, '?')
      dialogs[index]['dialog'][r_id]['question'] = text

  return dialogs


# method to use synonym and skip optional parts stochastically
def skip_and_replace_phrases(text):
  # for each text in [], replace it with '' with probability 0.5
  matches = re.findall('(\[[ \w]*\])', text)
  for match in matches:
    if random.uniform(0, 1) > 0.5:
      text = text.replace(match, '')
    else:
      text = text.replace(match, match[1:-1])

  # remove empty spaces, if any
  text = re.sub(' +', ' ', text)

  # search for synonyms, replace at uniformly random
  text = text.lower()
  for key, values in SYNONYM_KEYS:
    if key in text:
      text = text.replace(key, random.choice(values))

  return text


def generate_captions(scenes, templates):
  template_dictionary = {ii['label']: ii for ii in templates}
  generated_content = []
  for scene in scenes['scenes'][0:FLAGS.num_images]:
    content = {}
    # copy over image_index, split, image_filename from scene
    for key in ['image_index', 'split', 'image_filename']:
      content[key] = scene[key]

    content['dialogs'] = []

    # filter objects based on constraints
    filter_objs = constraints.caption(scene, templates)

    for filter_obj in filter_objs:
      # realize the text, and return the partial scene knowledge (q)
      template = template_dictionary[filter_obj[0]['graph']['template']]
      sample = realize_text_and_extract_scene(scene, template, filter_obj)
      # add it to the list of dialogs
      content['dialogs'].append(sample)

    generated_content.append(content)

  # Print generated content.
  # for index, datum in enumerate(generated_content):
  #   print('%d' % index)
  #   for ii in datum['dialogs']:
  #     print('\t' + ii['caption'])
  #     print(ii['graph'])

  return generated_content


def generate_captions_original(scenes, templates):
  generated_content = []
  for scene in scenes['scenes'][0:FLAGS.num_images]:
    content = {}
    # copy over image_index, split, image_filename from scene
    for key in ['image_index', 'split', 'image_filename']:
      content[key] = scene[key]

    content['dialogs'] = []
    cur_id = 0
    for cap_id in range(FLAGS.captions_per_image):
      flag = False

      while not flag:
        # pick a template at random
        template = random.choice(templates)
        #template = templates[cur_id % len(templates)]

        # filter objects based on constraints
        filter_objs = constraints.caption(scene, templates)
        flag = len(filter_objs) != 0

      # realize the text, and return the partial scene knowledge (q)
      sample = realize_text_and_extract_scene(scene, template, filter_objs)
      # add it to the list of dialogs
      content['dialogs'].append(sample)

    generated_content.append(content)

  # Print generated content.
  # for index, datum in enumerate(generated_content):
  #   print('%d' % index)
  #   for ii in datum['dialogs']:
  #     print('\t' + ii['caption'])
  #     print(ii['graph'])

  return generated_content


def generate_questions(scenes, dialogs, templates, params):
  new_dialogs = []
  for scene_id, dialog_datum in enumerate(dialogs):
    image_dialogs = copy.deepcopy(dialog_datum)
    image_dialogs['dialogs'] = []

    for dialog in dialog_datum['dialogs']:
      # pick a template at random
      flag = False
      iter_count = 0
      while not flag:
        # pick a template at random
        template = random.choice(templates)

        # filter objects based on constraints
        filter_objs = constraints.question(scenes['scenes'][scene_id],
                                           dialog, template)
        flag = len(filter_objs) != 0

        # extreme case -- exit
        iter_count += 1
        if iter_count > 10:
          break

      # realize q question
      if flag:
        deep_copy = copy.deepcopy(dialog)
        gen_dialog = realize_question(deep_copy, template, filter_objs)
        image_dialogs['dialogs'].append(copy.deepcopy(gen_dialog))

    new_dialogs.append(image_dialogs)
  return new_dialogs


# worker wrapper
def worker(scenes, cap_templates, ques_templates, worker_id, out_q):
  dialogs = []

  for index, scene in enumerate(scenes):
    cur_time = time.strftime('%a-%d%b%y-%X', time.gmtime())
    print('Generating [ %s ] [ Worker: %d, Progress: %d/%d Scene:  %d ]' % \
          (cur_time, worker_id, index, len(scenes), scene['image_index']))
    try:
      gen_dialog = generate_dialog_bfs(scene, cap_templates, ques_templates)
      dialogs.append(json.loads(json.dumps(gen_dialog)))
    except:
      print('NOTE: Missing data for %d' % scene['image_index'])

  out_q.put({worker_id: dialogs})


def generate_dialog_bfs(scene, cap_templates, ques_templates):
  bundle = {}
  # generate captions for the scene
  # copy over image_index, split, image_filename from scene
  for key in ['image_index', 'split', 'image_filename']:
    bundle[key] = scene[key]

  template_dictionary = {ii['label']: ii for ii in cap_templates}
  content = {}

  # filter objects based on constraints on captions
  filter_objs = constraints.caption(scene, cap_templates)

  for filter_obj in filter_objs:
    # realize the text, and return the partial scene knowledge (q)
    template = template_dictionary[filter_obj[0]['graph']['template']]
    sample = realize_text_and_extract_scene(scene, template, filter_obj)
    # add it to the list of dialogs
    content[template['label']] = [sample]

  # print generated content
  #for ii in content['dialogs']:
  #    print('\t' + ii['caption'])

  #---------------------------------------------
  # now questions
  # group templates, exist/count of similar type together
  ques_groups = collections.defaultdict(list)

  labels = [ii['label'] for ii in ques_templates]
  #print('\n'.join(labels))
  for index, ii in enumerate(ques_templates):
    if 'exist' in ii['label'] or 'count' in ii['label']:
      ques_groups[labels[index][4:]].append(ii)
    else:
      ques_groups[labels[index]].append(ii)

  #print('Threads at round %d: %d' % (-1, len(content['dialogs'])))
  for round_id in range(FLAGS.num_rounds):
    new_content = {}

    # for each group
    for cap_label, cap_dialogs in content.items():
      cur_pool = []
      for dialog_datum in cap_dialogs:
        for _, group in ques_groups.items():
          template = random.choice(group)
          # make a copy
          datum_copy = copy.deepcopy(dialog_datum)

          # filter objects based on constraints
          filter_objs = constraints.question(scene, datum_copy, template)
          if len(filter_objs) == 0: continue

          # realize q question
          gen_dialog = realize_question(datum_copy, template, filter_objs)
          cur_pool.append(gen_dialog)

      if round_id in ranges:
        for d_id, dialog in enumerate(cur_pool):
          n_types = {'indep': 0, 'seek': 0, 'exist': 0, 'count': 0}
          keep_dialog = True

          labels = [ii['label'] for ii in dialog['template_info'][1:]]
          for label in labels:
            if label in indep_ques_templates: n_types['indep'] += 1

            label_type = label.split('-')[0]
            n_types[label_type] += 1

          # Heuristic A, C
          for q_type, count in n_types.items():
            limit = ranges[round_id][q_type]
            if limit[0] > count or count > limit[1]:
              keep_dialog = False
              break

          # Heuristic B
          limit = ranges[round_id]['exist+count']
          if n_types['count'] + n_types['exist'] > limit[1]:
            keep_dialog = False

          #limit = ranges[round_id]['early']
          #count = [ii for ii in labels if 'early' in ii]
          #if limit[0] > count or count > limit[1]:
          #    keep_dialog = False
          #    break

          if not keep_dialog: cur_pool[d_id] = None

        cur_pool = [ii for ii in cur_pool if ii is not None]

      # keep limited number of beams (for speed)
      if len(cur_pool) > FLAGS.num_beams:
        cur_pool = sample_beams(cur_pool)[:FLAGS.num_beams]
      new_content[cap_label] = cur_pool

    content = copy.deepcopy(new_content)

  # get dialogs with sim, imm2, early questions
  for cap_label, cap_dialogs in content.items():
    # sample beams
    content[cap_label] = sample_beams(cap_dialogs)

  # remove keys that are empty
  empty_keys = [key for key, val in content.items() if len(val) == 0]
  for key in empty_keys:
    del content[key]

  # for each caption, sample one
  sampled_dialogs = []
  for cap_label, cap_dialogs in content.items():
    #if len(cap_dialogs) == 0: print('Empty cap dialog: %s' % cap_label)
    if len(cap_dialogs) > 0:
      sampled_dialogs.append(cap_dialogs.pop())

  # get 5 per image, compensate by taking from other entries
  content_keys = [ii for ii in content.keys()]
  while len(sampled_dialogs) < 5:
    random_label = random.choice(content_keys)
    sampled_dialogs.append(cap_dialogs.pop())

  # finally, make the dialog text readiable
  sampled_dialogs = clean_dialog_text(sampled_dialogs)

  # generate the coreference chain
  for dialog_id, dialog in enumerate(sampled_dialogs):
    sampled_dialogs[dialog_id] = identify_coref_chains(dialog)
  bundle['dialogs'] = sampled_dialogs

  return bundle


def sample_beams(dialogs):
  num_constraints = []
  for d_id, dialog in enumerate(dialogs):
    satisfied = 0
    labels = [ii['label'] for ii in dialog['template_info'][1:]]

    # have a imm2 for sure
    satisfied += np.sum(['imm2' in ii for ii in labels])
    # have a imm2 for sure
    satisfied += np.sum(['sim' in ii for ii in labels])
    # have 'early'
    satisfied += min(4, np.sum(['early' in ii for ii in labels]))

    # add it with the number of constraints it satisfies
    num_constraints.append((satisfied, d_id))

  # then order
  sort_key = lambda x: (x[0], random.random())
  ids = sorted(num_constraints, key=sort_key, reverse=True)
  return [dialogs[ii[1]] for ii in ids]


def identify_coref_chains(dialog):
  '''
  '''
  for r_id, datum in enumerate(dialog['dialog']):
    label = datum['template']
    if label in indep_ques_templates:
      dialog['graph']['history'][r_id + 1]['dependence'] = None
      continue

    if label == 'exist-attribute-group' or \
        label == 'count-attribute-group' or \
          label == 'count-all-group':
      dialog['graph']['history'][r_id + 1]['dependence'] = r_id-1
      continue

    if 'imm' in label:
      dialog['graph']['history'][r_id + 1]['dependence'] = r_id-1
      continue

    if 'early' in label:
      # go over previous history
      cur_history  = dialog['graph']['history'][r_id + 1]
      assert 'focus_id' in cur_history and 'focus_desc' in cur_history,\
       'More focus objects than one, no focus objects!'
      focus_id = cur_history['focus_id']
      for attr in attributes:
        if attr in cur_history['focus_desc']: break

      history = dialog['graph']['history'][:r_id+1]
      for hist_id, hist_datum in enumerate(history):
        for obj in hist_datum['objects']:
          if obj['id'] == focus_id and attr in obj:
            dialog['graph']['history'][r_id+1]['dependence'] = hist_id-1
            break

  return dialog
  # print for debugging
  #print(dialog['caption'])
  #for r_id, datum in enumerate(dialog['dialog']):
  #    label = datum['template']
  #    refer = dialog['graph']['history'][r_id + 1].get('dependence', None)
  #    print('\t_q-%d: %s [%s] [%s]' % (r_id, datum['question'], label, refer))
  #    print('\t_a-%d: %s' % (r_id, datum['answer']))


def main(_):
  # read scene file
  with open(FLAGS.scene_path, 'r') as file_id:
    scenes = json.load(file_id)

  # read the synonyms
  with open(FLAGS.synonym_path, 'r') as file_id:
    synonyms = json.load(file_id)
  sorter = lambda x: len(x[0].split(' '))
  global SYNONYM_KEYS
  SYNONYM_KEYS = sorted(synonyms.items(), key=sorter, reverse=True)

  # add ids to objects
  scenes = utils.add_object_ids(scenes)
  scenes = utils.clean_object_attributes(scenes)

  # Read caption templates.
  template_paths = os.listdir(FLAGS.caption_template_root)
  # debug_template = ['extreme_location.json',
  #                   'multiple_objects.json',
  #                   'object_relations.json',
  #                   'unique_object.json']
  # debug_templates = ['unique_object.json']
  cap_templates = []
  for ii in template_paths:
    # if ii not in debug_templates:
    #   # DEBUG
    #   continue
    with open(os.path.join(FLAGS.caption_template_root, ii), 'r') as file_id:
      cur_templates = json.load(file_id)
      cap_templates.extend(cur_templates)

  # utils.pretty_print_templates(cap_templates, 1)

  # DEBUG: generate captions
  # dialogs = generate_captions(scenes, cap_templates)
  # for dialog in dialogs:
  #   print('\n'.join([skip_and_replace_phrases(ii['caption'])
  #                    for ii in dialog['dialogs']]))
  #------------------------------------------------------------------------

  # Read question templates.
  template_paths = os.listdir(FLAGS.question_template_root)
  ques_templates = []
  debug_templates = ['count_question.json',
                     'exist_question.json',
                     'seek_attribute.json']
  for ii in template_paths:
    if ii not in debug_templates:
      continue

    with open(os.path.join(FLAGS.question_template_root, ii), 'r') as file_id:
     cur_templates = json.load(file_id)
     ques_templates.extend(cur_templates)
  #utils.pretty_print_templates(ques_templates, 1)

  # DEBUG: Run on just one image.
  # scenes_subset = scenes['scenes'][0: FLAGS.num_images]
  # worker(scenes_subset, cap_templates, ques_templates, 0, None)

  # 1. Check if there a scene_id_file specified.
  # 2. Check if num_images is -1
  if FLAGS.scene_id_file != '':
    with open(FLAGS.scene_id_file, 'r') as file_id:
      missing_ids = [int(ii.strip('\n')) for ii in file_id.readlines()]
    print('Dialogs missing for scenes: %d' % len(missing_ids))

    # Create a image_index -> scenes list index dictionary
    image_list_id_dict = {ii['image_index']: index
                          for index, ii in enumerate(scenes['scenes'])}
    scenes_subset = [scenes['scenes'][image_list_id_dict[scene_id]]
                     for scene_id in missing_ids]

  elif FLAGS.num_images == -1:
    scenes_subset = scenes['scenes']

  else:
    scenes_subset = scenes['scenes'][0: FLAGS.num_images]

  # BFS for each scene
  # single thread version
  # dialogs = []
  # for index, scene in enumerate(scenes_subset):
  #   cur_time = time.strftime('%a-%d%b%y-%X', time.gmtime())
  #   print('Generating [ %s ] [ Worker: %d, Progress: %d/%d Scene:  %d ]' %\
  #         (cur_time, 0, index, len(scenes_subset), scene['image_index']))
  #   gen_dialog = generate_dialog_bfs(scene, cap_templates, ques_templates)
  #   dialogs.append(gen_dialog)

  # multithread version
  output_q = multiprocessing.Queue()
  jobs = []
  for worker_id in range(FLAGS.num_workers):
    allotment = scenes_subset[worker_id::FLAGS.num_workers]
    inputs = (allotment, cap_templates, ques_templates)
    inputs += (worker_id, output_q)

    process = multiprocessing.Process(target=worker, args=inputs)
    jobs.append(process)
    process.start()

  # Wait for all the jobs to finish and collect.
  final_results = {}
  for _ in jobs:
    final_results.update(output_q.get())
  for job in jobs:
    job.join()

  # Flatten and sort.
  final_results = [jj for _, ii in final_results.items() for jj in ii]
  dialogs = sorted(final_results, key=lambda x: x['image_index'])

  # Generate the coreference chain.
  #for datum_id, datum in enumerate(dialogs):
  #  for dialog_id, dialog in enumerate(datum['dialogs']):
  #    dialogs[datum_id]['dialogs'][dialog_id] = identify_coref_chains(dialog)

  # utils.pretty_print_dialogs(dialogs)

  #for ii in range(10):
  #  dialogs = generate_questions(scenes, dialogs, templates, params)
  # save the dialogs
  print('Saving dialog at: %s' % FLAGS.save_path)
  with open(FLAGS.save_path, 'w') as file_id:
    json.dump(dialogs, file_id)


if __name__ == '__main__':
  app.run(main)
