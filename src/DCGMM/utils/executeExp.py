# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
""" Tool for executing multiple experiments based on a single bash file template.
Takes:   <bashFile> {<arg> <rep>}
Creates: a bashFile tmp.bash where every occurrence of <arg> has been updated by <rep>
Open:
- If <arg> does not exist it is ignored --> TODO should we allow this?
- assumes that params --STAAART and --EEEND are not used
""" 
import os, sys ;

def parseStr(tokens):
  tmpDict = {}
  # replace entries in bash file representation
  tokenIndex = 0 ;
  while True:
    token = tokens[tokenIndex] ;
    #print ("/"+token+"/")
    if token == "--EEEEEND": break ;
    if token.find("--")==0:
      tmp = tokenIndex+1 ;
      argstr = "";
      while tokens[tmp].find("--") != 0:
        argstr += " "+tokens[tmp] ;
        tmp += 1 ;
      tmpDict[token.strip()] = argstr.strip() ;
    tokenIndex += 1 ;
  return tmpDict ;

def createStrRepFromIterable(it):
  # construct dict of replacements
  argStr = "" ;
  for arg in it:
    argStr = argStr + " " + arg + " " ;
  args = ("--STAAART_ARGV "+argStr+" --EEEEEND").split(" ") ;
  return args ;


def createStrRepFromBashLines(lines):
  # construct dict representation of bash file
  str = "" ;

  for line in lines:
    if line[0] == "#": continue ;
    str = str + line.replace("\\", " ").replace("\n"," ") ;
  tokens = ("--STAAART "+str+" --EEEEEND").split(" ") ;
  return tokens ;


def createStrRepFromBashLine(line):
  return createStrRepFromBashLines([line]) ;


# create dict of replacements
args = createStrRepFromIterable(sys.argv[2:]) ;
repDict = parseStr(args)


# create dict of actual params in bash file
bashfile = sys.argv[1] ;
lines = open(bashfile,"r").readlines() ;

tokens = createStrRepFromBashLines(lines) ;
bashDict = parseStr(tokens) ;

for repKey,repVal in repDict.items():
  print (f"replacing /{repKey}/")
  bashDict[repKey] = repVal ;

#print ("bash", bashDict)


tmpBashFile = open("tmp.bash","w") ;
tmpBashFile.write(bashDict["--STAAART"].strip()+" \\\n")
for bashKey,bashVal in bashDict.items():
  if bashKey == "--STAAART": continue ;
  tmpBashFile.write(bashKey+" "+bashVal+" \\\n") ;
tmpBashFile.close() ;
