/*
 * $Id$
 */

#ifndef __POSTAGGER_COMMON_H_
#define __POSTAGGER_COMMON_H_

#include <string>
#include <vector>
#include <map>

struct Token
{
  //  std::string org_str;
  std::string str;
  std::string pos;
  std::string prd;
  int begin;
  int end;
  Token(std::string s, std::string p) : str(s), pos(p) {}
  Token(std::string s, const int b, const int e) : str(s), begin(b), end(e) {}
};


class Sentence
{
    public:
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

        std::vector<Token>::iterator begin()
        {
            return content.begin();
        }

        std::vector<Token>::iterator end()
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


#endif

/*
 * $Log$
 */
