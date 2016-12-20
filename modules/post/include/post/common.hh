/*
 * $Id$
 */

#ifndef NLPTOOLS_POST_COMMON_HH
#define NLPTOOLS_POST_COMMON_HH

#include <string>
#include <vector>
#include <map>

namespace nlptools {
namespace postagger {


struct Token
{
  //  std::string org_str;
  std::string str; // token string
  std::string pos; // token tag (golden)
  std::string prd; // token tag (predicted)
  int begin;
  int end;
  Token(std::string s, std::string p) : str(s), pos(p) {}
  Token(std::string s, const int b, const int e) : str(s), begin(b), end(e) {}
};


class Sentence
{
    public:
        typedef std::vector<Token>::iterator iterator;
        typedef std::vector<Token>::const_iterator const_iterator;

        Sentence()
        {
        }

        ~Sentence()
        {
        }

        Token &operator [](
            size_t index )
        {
            return content[index];
        }

        const Token &operator [](
            size_t index ) const
        {
            return content[index];
        }

        size_t size() const
        {
            return content.size();
        }

        void push_back(
            const Token &item )
        {
            content.push_back(item);
        }

        void pop_back()
        {
            content.pop_back();
        }

        iterator begin()
        {
            return content.begin();
        }

        iterator end()
        {
            return content.end();
        }

        const_iterator begin() const
        {
            return content.begin();
        }

        const_iterator end() const
        {
            return content.end();
        }

    private:
        std::vector<Token> content;
};



class ParenConverter
{
  std::map<std::string, std::string> ptb2pos;
  std::map<std::string, std::string> pos2ptb;
public:
  ParenConverter() {
    const static char* table[] = {
      "-LRB-", "(",
      "-RRB-", ")",
      "-LSB-", "[",
      "-RSB-", "]",
      "-LCB-", "{",
      "-RCB-", "}",
      "***", "***",
    };

    for (int i = 0;; i+=2) {
      if (std::string(table[i]) == "***") break;
      ptb2pos.insert(std::make_pair(table[i], table[i+1]));
      pos2ptb.insert(std::make_pair(table[i+1], table[i]));
    }
  }
  std::string Ptb2Pos(const std::string & s) {
    std::map<std::string, std::string>::const_iterator i = ptb2pos.find(s);
    if (i == ptb2pos.end()) return s;
    return i->second;
  }
  std::string Pos2Ptb(const std::string & s) {
    std::map<std::string, std::string>::const_iterator i = pos2ptb.find(s);
    if (i == pos2ptb.end()) return s;
    return i->second;
  }
};


}}


#endif // NLPTOOLS_POST_COMMON_HH
