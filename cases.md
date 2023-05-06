##### An example of correctly identified libraries
`CVE-2022-2963` is an example vulnerability that VulLibMiner correctly recommand its affected libraries.
It is a SQL injection vulnerability whose description is listed as follows:

It is a memory leak bug of jasper whose description is listed as follows:
~~~
A vulnerability found in jasper. This security vulnerability happens because of a memory leak bug in function cmdopts_parse that can cause a crash or segmentation fault.
~~~

However, NVD only includes `cpe:2.3:a:jasper_project:jasper:3.0.6:` as its affected CPE, along with `fedora` and `linux`.
VulLibMiner recommand two `jasper-compiler` libraries that also includes the vulnerable function `cmdopts_parse`.

~~~
maven:jetty:jasper-compiler
maven:tomcat:jasper-compiler
maven:tomcat:jasper
~~~

##### An example of incorrectly identified libraries
`CVE-2022-3471` is an example vulnerability that VulLibMiner incorrectly recommand its affected libraries.
It is a SQL injection vulnerability whose description is listed as follows:
```
A vulnerability was found in SourceCodester Human Resource Management System. It has been declared as critical. Affected by this vulnerability is an unknown functionality of the file city.php. The manipulation of the argument searccity leads to sql injection. The attack can be launched remotely. The exploit has been disclosed to the public and may be used. The associated identifier of this vulnerability is VDB-210715.
```
Actually, it is a php vulnerability because `SourceCodester Human Resource Management System` is written in php.
However, the description of this vulnerability does not clearly tell the affected system and how this vulnerability is utilized.
Thus, VulLibMiner outputs some sql libraries and php libraries.
~~~
maven:org.apache.storm:sql
maven:cn.sexycode:my-sql
maven:dev.rudiments:sql
maven:xyz.cofe:sql
maven:taglibs:sql
maven:commons-sql:commons-sql
maven:org.cloudhoist:php
maven:cocoon:cocoon-php
~~~


##### Conclusion
From the preceding two examples, we show that VulLibMiner can identify vulnerable libraries (especially those neglected in NVD) while its effectiveness might be affected by the quality of vulnerability descriptions.

