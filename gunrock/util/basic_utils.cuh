
namespace gunrock {
namespace util {

	template<typename T> struct make_unsigned;
	
	template<> struct make_unsigned<char>                   { typedef unsigned char          type; };
	template<> struct make_unsigned<signed char>            { typedef signed   char          type; };
	template<> struct make_unsigned<unsigned char>          { typedef unsigned char          type; };
	template<> struct make_unsigned<short>                  { typedef unsigned short         type; };
	template<> struct make_unsigned<unsigned short>         { typedef unsigned short         type; };
	template<> struct make_unsigned<int>                    { typedef unsigned int           type; };
	template<> struct make_unsigned<unsigned int>           { typedef unsigned int           type; };
	template<> struct make_unsigned<long int>               { typedef unsigned long int      type; };
	template<> struct make_unsigned<unsigned long int>      { typedef unsigned long int      type; };
	template<> struct make_unsigned<long long int>          { typedef unsigned long long int type; };
	template<> struct make_unsigned<unsigned long long int> { typedef unsigned long long int type; };
}
}
